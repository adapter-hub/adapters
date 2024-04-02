import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    ImageClassifierOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.utils import ModelOutput

from ..composition import adjust_tensors_for_parallel
from ..methods.modeling import Activation_Function_Class


logger = logging.getLogger(__name__)


@dataclass
class MultiHeadOutput(ModelOutput):
    head_outputs: List[ModelOutput] = None
    loss: Optional[torch.FloatTensor] = None

    @property
    def logits(self):
        return torch.vstack([outputs["logits"] for outputs in self.head_outputs])

    def __getitem__(self, k):
        # with number indices the head output at that position is accessed
        # e.g output[1] is equivalent to output.head_outputs[1]
        if isinstance(k, int):
            return self.head_outputs[k]
        # with strings the attribute in the underlying dict can be adressed
        # e.g output["loss"] is equivalent to output.loss
        else:
            return super().__getitem__(k)

    def __setitem__(self, k, v):
        if isinstance(k, int):
            self.head_outputs[k] = v
        else:
            return super().__setitem__(k, v)

    def __iter__(self):
        # iterates over the head outputs
        return iter(self.head_outputs)

    def __len__(self):
        return len(self.head_outputs)


# Let this class inherit from nn.Sequential to provide iterable access as before
class PredictionHead(nn.Sequential):
    def __init__(self, name):
        super().__init__()
        self.config = {}
        self.name = name

    def build(self, model):
        model_config = model.config
        pred_head = []
        if "dropout_prob" in self.config and self.config["dropout_prob"] is not None:
            dropout_prob = self.config["dropout_prob"]
        elif hasattr(model_config, "classifier_dropout") and model_config.classifier_dropout is not None:
            dropout_prob = model_config.classifier_dropout
        else:
            dropout_prob = model_config.hidden_dropout_prob
        bias = self.config.get("bias", True)
        for l_id in range(self.config["layers"]):
            pred_head.append(nn.Dropout(dropout_prob))
            if l_id < self.config["layers"] - 1:
                pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size))
                if self.config["activation_function"]:
                    pred_head.append(Activation_Function_Class(self.config["activation_function"]))
            else:
                if "num_labels" in self.config:
                    pred_head.append(nn.Linear(model_config.hidden_size, self.config["num_labels"], bias=bias))
                elif "num_choices" in self.config:  # used for multiple_choice head
                    pred_head.append(nn.Linear(model_config.hidden_size, 1, bias=bias))
                else:
                    pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=bias))
                    if self.config["activation_function"]:
                        pred_head.append(Activation_Function_Class(self.config["activation_function"]))
        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        # We need to import the current value of _init_weights at each execution to determine if weights init is disabled.
        from transformers.modeling_utils import _init_weights

        if _init_weights:
            self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent

    def get_output_embeddings(self):
        return None  # override for heads with output embeddings

    def get_label_names(self):
        return ["labels"]

    def _get_cls_output(self, outputs, **kwargs):
        if self.config["use_pooler"]:
            cls_output = kwargs.pop("pooled_output")
        elif kwargs.get("get_cls_from_eos_tokens", False):
            x = outputs[0]  # last hidden state
            eos_mask = kwargs.get("eos_mask")
            (eos_mask,) = adjust_tensors_for_parallel(x, eos_mask)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            cls_output = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        else:
            cls_output = outputs[0][:, 0]

        return cls_output


class ClassificationHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        id2label=None,
        use_pooler=False,
        bias=True,
        dropout_prob=None,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "use_pooler": use_pooler,
            "bias": bias,
            "dropout_prob": dropout_prob,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        if cls_output is None:
            cls_output = self._get_cls_output(outputs, **kwargs)
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            if self.config["num_labels"] == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config["num_labels"]), labels.view(-1))

        if return_dict:
            if isinstance(outputs, Seq2SeqModelOutput):
                return Seq2SeqSequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )
            else:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs


class MultiLabelClassificationHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        id2label=None,
        use_pooler=False,
        bias=True,
        dropout_prob=None,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "multilabel_classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "use_pooler": use_pooler,
            "bias": bias,
            "dropout_prob": dropout_prob,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        if cls_output is None:
            cls_output = self._get_cls_output(outputs, **kwargs)
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            if labels.dtype != torch.float32:
                labels = labels.float()
            loss = loss_fct(logits, labels)

        if return_dict:
            if isinstance(outputs, Seq2SeqModelOutput):
                return Seq2SeqSequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )
            else:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs


class MultipleChoiceHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_choices=2,
        layers=2,
        activation_function="tanh",
        id2label=None,
        use_pooler=False,
        dropout_prob=None,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "multiple_choice",
            "num_choices": num_choices,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "use_pooler": use_pooler,
            "dropout_prob": dropout_prob,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=None, **kwargs):
        if cls_output is None:
            cls_output = self._get_cls_output(outputs, **kwargs)
        logits = super().forward(cls_output)
        logits = logits.view(-1, self.config["num_choices"])
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if return_dict:
            return MultipleChoiceModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs


class TaggingHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=1,
        activation_function="tanh",
        id2label=None,
        dropout_prob=None,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "tagging",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "dropout_prob": dropout_prob,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        logits = super().forward(outputs[0])
        loss = None

        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # adjust labels for prompt tuning
            if kwargs.get("prompt_tokens_length", 0) > 0:
                prompt_length = kwargs.get("prompt_tokens_length")
                prompt_labels = torch.full(
                    (labels.shape[0], prompt_length), loss_fct.ignore_index, dtype=torch.long, device=labels.device
                )
                labels = torch.cat((prompt_labels, labels), dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        (torch.ones_like(prompt_labels, dtype=torch.long, device=labels.device), attention_mask),
                        dim=-1,
                    )

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config["num_labels"])
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config["num_labels"]), labels.view(-1))

        if return_dict:
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs


class QuestionAnsweringHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=1,
        activation_function="tanh",
        id2label=None,
        dropout_prob=None,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "question_answering",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "dropout_prob": dropout_prob,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        sequence_output = outputs[0]
        logits = super().forward(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_positions = kwargs.pop("start_positions", None)
        end_positions = kwargs.pop("end_positions", None)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if return_dict:
            if isinstance(outputs, Seq2SeqModelOutput):
                return Seq2SeqQuestionAnsweringModelOutput(
                    loss=total_loss,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )
            else:
                return QuestionAnsweringModelOutput(
                    loss=total_loss,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            outputs = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            if total_loss is not None:
                outputs = (total_loss,) + outputs
            return outputs

    def get_label_names(self):
        return ["start_positions", "end_positions"]


class ImageClassificationHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        multilabel=False,
        id2label=None,
        use_pooler=False,
        bias=True,
        dropout_prob=None,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "image_classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "multilabel": multilabel,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "use_pooler": use_pooler,
            "bias": bias,
            "dropout_prob": dropout_prob,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        if cls_output is None:
            cls_output = self._get_cls_output(outputs, **kwargs)
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            if self.config["num_labels"] == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.config["multilabel"]:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config["num_labels"]), labels.view(-1))

        if return_dict:
            return ImageClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs
