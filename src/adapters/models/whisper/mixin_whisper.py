from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperForCausalLM, WhisperForAudioClassification

from ...composition import adjust_tensors_for_parallel
from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import (
    EmbeddingAdaptersMixin,
    EmbeddingAdaptersWrapperMixin,
    InvertibleAdaptersMixin,
    InvertibleAdaptersWrapperMixin,
    ModelBaseAdaptersMixin,
    ModelWithHeadsAdaptersMixin,
)


class WhisperAttentionAdaptersMixin:
    """Adds adapters to the WhisperAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.k_proj = LoRALinear.wrap(self.k_proj, "selfattn", model_config, adapters_config, attn_key="k")
        self.v_proj = LoRALinear.wrap(self.v_proj, "selfattn", model_config, adapters_config, attn_key="v")
        self.q_proj = LoRALinear.wrap(self.q_proj, "selfattn", model_config, adapters_config, attn_key="q")

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )


class WhisperEncoderLayerAdaptersMixin:
    """Adds adapters to the WhisperEncoderLayer module of WHISPER."""

    def init_adapters(self, model_config, adapters_config):
        self.adapters_config = adapters_config
        # Wrap layers for LoRA
        self.fc1 = LoRALinear.wrap(self.fc1, "intermediate", model_config, adapters_config)
        self.fc2 = LoRALinear.wrap(self.fc2, "output", model_config, adapters_config)

        # Set attention layer location key for prefix tuning
        self.self_attn.location_key = "encoder"
        self.attention_adapters = BottleneckLayer("mh_adapter")
        self.output_adapters = BottleneckLayer("output_adapter")


class WhisperDecoderLayerAdaptersMixin(WhisperEncoderLayerAdaptersMixin):
    """Adds adapters to the WhisperDecoderLayer module of WHISPER."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)
        # Set attention layer location key for prefix tuning
        self.self_attn.location_key = "self"
        self.encoder_attn.location_key = "cross"
        self.cross_attention_adapters = BottleneckLayer("cross_adapter")


class WhisperEncoderAdaptersMixin(InvertibleAdaptersMixin):
    """Adds adapters to the WhisperEncoder module of WHISPER."""

    pass


class WhisperDecoderAdaptersMixin:
    """Adds adapters to the WhisperDecoder module of WHISPER."""

    def forward(
            self, input_ids: torch.LongTensor = None, encoder_hidden_states: Optional[torch.FloatTensor] = None,
            **kwargs
    ):
        (input_ids,) = adjust_tensors_for_parallel(encoder_hidden_states, input_ids)
        return super().forward(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, **kwargs)


class WhisperModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersWrapperMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the WhisperModel class."""

    invertible_adapters_base_name = "encoder"
    support_prompt_tuning = False

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        # Due to different model object structures, we need to handle the different model types separately
        # Conditional generation and Causal LM have the actual model object as an attribute
        if isinstance(self, WhisperForConditionalGeneration):
            for i, layer in enumerate(self.model.encoder.layers):
                yield i, layer
            for i, layer in enumerate(self.model.decoder.layers, start=len(self.model.encoder.layers)):
                yield i, layer
        if isinstance(self, WhisperForCausalLM):
            for i, layer in enumerate(self.model.decoder.layers):
                yield i, layer
        # Audio classification model object in comparison is the actual model
        if isinstance(self, WhisperForAudioClassification):
            for i, layer in enumerate(self.encoder.layers):
                yield i, layer


class WhisperDecoderWrapperAdaptersMixin(EmbeddingAdaptersWrapperMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the WhisperDecoderWrapper class.

    This wrapper class is a helper class to correctly load
    pretrained checkpoints when the causal language model is used in combination with the [`EncoderDecoderModel`]
    framework."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.decoder.layers):
            yield i, layer

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()


class WhisperForAudioClassificationWithHeadsMixin(WhisperModelAdaptersMixin, ModelWithHeadsAdaptersMixin):
    """Adds adapters to the WhisperForAudioClassification class."""

    def forward(self, input_features: Optional[torch.FloatTensor] = None, **kwargs):
        return super().forward(input_features=input_features, **kwargs)


class WhisperForConditionalGenerationWithHeadsMixin(WhisperModelAdaptersMixin, ModelWithHeadsAdaptersMixin):
    """Adds adapters to the WhisperForConditionalGeneration class."""

    def forward(self, input_features: torch.FloatTensor, **kwargs):
        return super().forward(input_features=input_features, **kwargs)


class WhisperForCausalLMWithHeadsMixin(WhisperModelAdaptersMixin, ModelWithHeadsAdaptersMixin):
    """Adds adapters to the WhisperForCausalLM class."""

    def forward(self, input_features: torch.FloatTensor, **kwargs):
        return super().forward(input_features=input_features, **kwargs)
