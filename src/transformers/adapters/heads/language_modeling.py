import torch.nn as nn

from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, Seq2SeqLMOutput
from .base import PredictionHead


class CausalLMHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        shift_labels=True,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "causal_lm",
            "num_labels": model.config.vocab_size,
            "layers": 1,
            "activation_function": None,
            "dropout_prob": 0,
            "bias": False,
            "shift_labels": shift_labels,
        }
        self.build(model)

    def get_output_embeddings(self):
        # The last child is our embedding layer
        return self._modules[next(reversed(self._modules))]

    def set_output_embeddings(self, new_embeddings):
        # The last child is our embedding layer
        self._modules[next(reversed(self._modules))] = new_embeddings

    @staticmethod
    def _create_model_output(loss, logits, base_outputs):
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        lm_logits = super().forward(outputs[0])

        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if self.config["shift_labels"]:
                logits_for_loss = lm_logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            else:
                logits_for_loss = lm_logits
            loss = loss_fct(logits_for_loss.view(-1, self.config["num_labels"]), labels.view(-1))

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
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "seq2seq_lm",
            "num_labels": model.config.vocab_size,
            "layers": 1,
            "activation_function": None,
            "dropout_prob": 0,
            "bias": False,
            "shift_labels": False,
        }
        self.build(model)

    @staticmethod
    def _create_model_output(self, loss, logits, base_outputs):
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
        activation_function="gelu",
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "masked_lm",
            "num_labels": model.config.vocab_size,
            "layers": 2,
            "activation_function": activation_function,
            "dropout_prob": 0,
            "layer_norm": True,
            "bias": False,
            "shift_labels": False,
        }
        self.build(model)

    @staticmethod
    def _create_model_output(self, loss, logits, base_outputs):
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )


class BertStyleCausalLMHead(CausalLMHead):
    def __init__(
        self,
        model,
        head_name,
        activation_function="gelu",
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "causal_lm",
            "num_labels": model.config.vocab_size,
            "layers": 2,
            "activation_function": activation_function,
            "dropout_prob": 0,
            "layer_norm": True,
            "bias": False,
            "shift_labels": True,
        }
        self.build(model)
