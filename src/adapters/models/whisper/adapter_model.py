import torch

from transformers import EncoderDecoderCache, StaticCache
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
    _tied_weights_keys = []
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
        **kwargs,
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
                decoder_input_ids = shift_tokens_right(
                    kwargs["labels"], self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Future TODO:
        # the seq2seqtrainer has the parameter `predict_with_generate`
        # If set to True, we get the following error:
        # transformers\generation\utils.py", line 1130, in _validate_model_kwargs">
        # raise ValueError(ValueError: The following model_kwargs are not used by the model: ['labels']
        # This is because we do not specify labels as parameter in the forward method

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

    # Copied from WhisperForConditionalGeneration
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            if decoder_position_ids is not None:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                decoder_position_ids = decoder_position_ids.clone(memory_format=torch.contiguous_format)

        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + decoder_input_ids.shape[1], device=decoder_input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-decoder_input_ids.shape[1] :]

        # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
        # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
        decoder_input_ids = decoder_input_ids.contiguous()

        if (
            isinstance(past_key_values, EncoderDecoderCache)
            and (
                isinstance(past_key_values.self_attention_cache, StaticCache)
                or isinstance(past_key_values.cross_attention_cache, StaticCache)
            )
            and decoder_attention_mask is not None
            and decoder_attention_mask.ndim == 2
        ):
            batch_size, sequence_length = decoder_input_ids.shape
            device = decoder_input_ids.device

            dtype = self.proj_out.weight.dtype
            min_dtype = torch.finfo(dtype).min

            decoder_attention_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                decoder_attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.self_attention_cache.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "cache_position": cache_position,
            # >>> START AH Changes <<<
            "adapter_input_parallelized": kwargs.pop("adapter_input_parallelized", False),
            # >>> END AH Changes <<<
        }

    # Copied from WhisperForConditionalGeneration
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
