import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .adapter_modeling import Activation_Function_Class
from .modeling_outputs import (
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


# Let this class inherit from nn.Sequential to provide iterable access as before
class PredictionHead(nn.Sequential):
    def __init__(self, name):
        super().__init__()
        self.config = {}
        self.name = name

    def build(self, model):
        model_config = model.config
        pred_head = []
        for l in range(self.config["layers"]):
            pred_head.append(nn.Dropout(model_config.hidden_dropout_prob))
            if l < self.config["layers"] - 1:
                pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size))
                pred_head.append(Activation_Function_Class(self.config["activation_function"]))
            else:
                if "num_labels" in self.config:
                    pred_head.append(nn.Linear(model_config.hidden_size, self.config["num_labels"]))
                else:  # used for multiple_choice head
                    pred_head.append(nn.Linear(model_config.hidden_size, 1))
        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent


class ClassificationHead(PredictionHead):
    def __init__(self, head_name, num_labels, layers, activation_function, id2label, model):
        super().__init__(head_name)
        self.config = {
            "head_type": "classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.build(model)

    def forward(self, outputs, attention_mask, return_dict, **kwargs):

        logits = super().forward(outputs[0][:, 0])
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
    def __init__(self, head_name, num_labels, layers, activation_function, id2label, model):
        super().__init__(head_name)
        self.config = {
            "head_type": "multilabel_classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.build(model)

    def forward(self, outputs, attention_mask, return_dict, **kwargs):
        logits = super().forward(outputs[0][:, 0])
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            if labels.dtype != torch.float32:
                labels = labels.float()
            loss = loss_fct(logits, labels)

        if return_dict:
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
    def __init__(self, head_name, num_choices, layers, activation_function, id2label, model):
        super().__init__(head_name)
        self.config = {
            "head_type": "multiple_choice",
            "num_choices": num_choices,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.build(model)

    def forward(self, outputs, attention_mask, return_dict, **kwargs):
        logits = super().forward(outputs[0][:, 0])
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
    def __init__(self, head_name, num_labels, layers, activation_function, id2label, model):
        super().__init__(head_name)
        self.config = {
            "head_type": "tagging",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.build(model)

    def forward(self, outputs, attention_mask, return_dict, **kwargs):
        logits = super().forward(outputs[0])
        loss = None

        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
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
    def __init__(self, head_name, num_labels, layers, activation_function, id2label, model):
        super().__init__(head_name)
        self.config = {
            "head_type": "question_answering",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.build(model)

    def forward(self, outputs, attention_mask, return_dict, **kwargs):
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
