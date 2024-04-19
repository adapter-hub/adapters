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
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    head_types = [
        "seq2seq_lm",
        "classification",
    ]

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

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
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
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            past_key_values=None,
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

        if "labels" in kwargs or "start_positions" in kwargs and "end_positions" in kwargs:
            use_cache = False

        outputs, context = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
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
            eos_mask=input_ids.eq(self.config.eos_token_id) if input_ids is not None else None,
            **kwargs,
        )

        return head_outputs
