import torch

from transformers.models.whisper.modeling_whisper import (
    WHISPER_INPUTS_DOCSTRING,
    WHISPER_START_DOCSTRING,
    WhisperConfig,
    WhisperModel,
    WhisperPreTrainedModel,
    shift_tokens_right,
)
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward

from ...heads import ModelWithFlexibleHeadsAdaptersMixin
from ...model_mixin import EmbeddingAdaptersWrapperMixin
from ...wrappers import init


@add_start_docstrings(
    "WHISPER Model with the option to add multiple flexible prediction heads on top.", WHISPER_START_DOCSTRING
)
class WhisperAdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, WhisperPreTrainedModel):
    head_types = ["seq2seq_lm"]

    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.model = WhisperModel(config)
        init(self.model)

        self._init_head_modules()

        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        # Adapted from WhisperModel in transformers/models/whisper/modeling_whisper.py as the tests in
        # test_modeling_whisper.py require the functionality to freeze the encoder
        self.model.encoder._freeze_parameters()

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head=None,
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if "labels" in kwargs:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels=kwargs["labels"])

        outputs, context = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
            output_adapter_gating_scores=output_adapter_gating_scores,
            output_adapter_fusion_attentions=output_adapter_fusion_attentions,
            adapter_input_parallelized=kwargs.pop("adapter_input_parallelized", False),
            output_context=True,
        )

        # required e.g. for prompt tuning in all models
        kwargs["context"] = context

        head_outputs = self.forward_head(
            outputs,
            head_name=head,
            attention_mask=attention_mask,
            return_dict=return_dict,
            get_cls_from_eos_tokens=True,
            # `get_cls_from_eos_tokens` requires passing eos mask
            eos_mask=input_features.eq(self.config.eos_token_id) if input_features is not None else None,
            **kwargs,
        )

        return head_outputs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # Copied from WhisperForConditionalGeneration
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": None,
            "adapter_input_parallelized": kwargs.pop("adapter_input_parallelized", False),
        }

    # Copied from WhisperForConditionalGeneration
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
