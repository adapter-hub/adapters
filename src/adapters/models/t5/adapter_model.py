import logging

import torch

from transformers.models.t5.modeling_t5 import T5_INPUTS_DOCSTRING, T5_START_DOCSTRING, T5Model, T5PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward

from ...composition import adjust_tensors_for_parallel
from ...heads import ModelWithFlexibleHeadsAdaptersMixin, Seq2SeqLMHead
from ...model_mixin import EmbeddingAdaptersWrapperMixin
from ...wrappers import init


logger = logging.getLogger(__name__)


@add_start_docstrings("T5 Model with the option to add multiple flexible prediction heads on top.", T5_START_DOCSTRING)
class T5AdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, T5PreTrainedModel):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    head_types = [
        "classification",
        "multilabel_classification",
        "question_answering",
        "seq2seq_lm",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.transformer = T5Model(config)
        init(self.transformer)

        self._init_head_modules()

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_encoder(self):
        return self.transformer.encoder

    def get_decoder(self):
        return self.transformer.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head=None,
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            # Check if we're using a LM head
            if labels is not None and any([isinstance(head, Seq2SeqLMHead) for head in self._get_used_heads(head)]):
                # get decoder inputs from shifting lm labels to the right
                decoder_input_ids = self._shift_right(labels)
            else:
                # decoder_input_ids from input_ids if no decoder_input_ids are provided
                decoder_input_ids = self._shift_right(input_ids)

        model_output, context = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_adapter_gating_scores=output_adapter_gating_scores,
            output_adapter_fusion_attentions=output_adapter_fusion_attentions,
            adapter_input_parallelized=kwargs.pop("adapter_input_parallelized", False),
            output_context=True,
        )
        # required e.g. for prompt tuning in all models
        kwargs["context"] = context
        sequence_output = model_output[0]
        # ToDo move head to device for parallel forward pass

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            new_hidden_state = sequence_output * (self.config.d_model**-0.5)
            if isinstance(model_output, tuple):
                model_output = (new_hidden_state,) + model_output[1:]
            else:
                model_output["last_hidden_state"] = new_hidden_state

        # sequence classification based on last token in sequence
        if input_ids is not None and sequence_output.shape[1] == input_ids.shape[1]:
            eos_mask = input_ids.eq(self.config.eos_token_id)
            (eos_mask,) = adjust_tensors_for_parallel(sequence_output, eos_mask)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            cls_representation = sequence_output[eos_mask, :].view(
                sequence_output.size(0), -1, sequence_output.size(-1)
            )[:, -1, :]
        else:
            cls_representation = sequence_output

        kwargs["labels"] = labels
        head_outputs = self.forward_head(
            model_output,
            head_name=head,
            cls_output=cls_representation,
            return_dict=return_dict,
            **kwargs,
        )
        return head_outputs

    # Copied from T5ForConditionalGeneration
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "adapter_input_parallelized": kwargs.pop("adapter_input_parallelized", False),
        }

    # Copied from T5ForConditionalGeneration
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    # Copied from T5ForConditionalGeneration
    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
