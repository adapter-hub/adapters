import torch.nn as nn

from ...modeling_outputs import CausalLMOutput, CausalLMOutputWithPast, MaskedLMOutput, Seq2SeqLMOutput
from ..modeling import Activation_Function_Class
from .base import PredictionHead


class CausalLMHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        vocab_size=None,
        layers=1,
        activation_function=None,
        layer_norm=False,
        bias=False,
        shift_labels=True,
    ):
        super(CausalLMHead, self).__init__(head_name)
        self.config = {
            "head_type": "causal_lm",
            "vocab_size": vocab_size or model.config.vocab_size,
            "layers": layers,
            "activation_function": activation_function,
            "layer_norm": layer_norm,
            "bias": bias,
            "shift_labels": shift_labels,
            "label2id": None,
        }
        self.build(model)

    def build(self, model):
        model_config = model.config
        # Additional FC layers
        pred_head = []
        with_layer_norm = self.config.get("layer_norm", False)
        for l_id in range(self.config["layers"] - 1):
            pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size))
            if self.config["activation_function"]:
                pred_head.append(Activation_Function_Class(self.config["activation_function"]))
            if with_layer_norm:
                eps = getattr(model_config, "layer_norm_eps", 1e-12)
                pred_head.append(nn.LayerNorm(model_config.hidden_size, eps=eps))
        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        # Final embedding layer
        self.add_module(
            str(len(pred_head)),
            nn.Linear(model_config.hidden_size, self.config["vocab_size"], bias=self.config["bias"]),
        )

        self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent

    def get_output_embeddings(self):
        # The last child is our embedding layer
        return self._modules[next(reversed(self._modules))]

    def set_output_embeddings(self, new_embeddings):
        # The last child is our embedding layer
        self._modules[next(reversed(self._modules))] = new_embeddings

    @staticmethod
    def _create_model_output(loss, logits, base_outputs):
        if "past_key_values" in base_outputs:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                hidden_states=base_outputs.hidden_states,
                attentions=base_outputs.attentions,
                past_key_values=base_outputs.past_key_values,
            )
        else:
            return CausalLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=base_outputs.hidden_states,
                attentions=base_outputs.attentions,
            )

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        # First, pass through all layers except the last embedding layer
        seq_outputs = outputs[0]
        for i in range(len(self) - 1):
            seq_outputs = self[i](seq_outputs)

        # Now, pass through an invertible adapter if available
        inv_adapter = kwargs.pop("invertible_adapter", None)
        if inv_adapter is not None:
            seq_outputs = inv_adapter(seq_outputs, rev=True)

        # Finally, pass through the last embedding layer
        lm_logits = self[len(self) - 1](seq_outputs)

        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if self.config["shift_labels"]:
                logits_for_loss = lm_logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            else:
                logits_for_loss = lm_logits
            loss = loss_fct(logits_for_loss.view(-1, self.config["vocab_size"]), labels.view(-1))

        if return_dict:
            return self._create_model_output(loss, lm_logits, outputs)
        else:
            outputs = (lm_logits,) + outputs[1:]
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs


class Seq2SeqLMHead(CausalLMHead):
    def __init__(
        self,
        model,
        head_name,
        vocab_size=None,
        layers=1,
        activation_function=None,
        layer_norm=False,
        bias=False,
        shift_labels=False,
    ):
        super(CausalLMHead, self).__init__(head_name)
        self.config = {
            "head_type": "seq2seq_lm",
            "vocab_size": vocab_size or model.config.vocab_size,
            "layers": layers,
            "activation_function": activation_function,
            "layer_norm": layer_norm,
            "bias": bias,
            "shift_labels": shift_labels,
            "label2id": None,
        }
        self.build(model)

    @staticmethod
    def _create_model_output(loss, logits, base_outputs):
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=base_outputs.past_key_values,
            decoder_hidden_states=base_outputs.decoder_hidden_states,
            decoder_attentions=base_outputs.decoder_attentions,
            cross_attentions=base_outputs.cross_attentions,
            encoder_last_hidden_state=base_outputs.encoder_last_hidden_state,
            encoder_hidden_states=base_outputs.encoder_hidden_states,
            encoder_attentions=base_outputs.encoder_attentions,
        )


class BertStyleMaskedLMHead(CausalLMHead):
    def __init__(
        self,
        model,
        head_name,
        vocab_size=None,
        layers=2,
        activation_function="gelu",
        layer_norm=True,
        bias=True,
        shift_labels=False,
    ):
        super(CausalLMHead, self).__init__(head_name)
        self.config = {
            "head_type": "masked_lm",
            "vocab_size": vocab_size or model.config.vocab_size,
            "layers": layers,
            "activation_function": activation_function,
            "layer_norm": layer_norm,
            "bias": bias,
            "shift_labels": shift_labels,
            "label2id": None,
        }
        self.build(model)

    @staticmethod
    def _create_model_output(loss, logits, base_outputs):
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )
