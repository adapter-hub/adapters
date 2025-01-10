from typing import Iterable, Tuple

import torch.nn as nn

from ...composition import adjust_tensors_for_parallel_
from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...methods.reft import ReftLayer, hook_fn
from ...model_mixin import (
    EmbeddingAdaptersMixin,
    InvertibleAdaptersMixin,
    ModelBaseAdaptersMixin,
)
from ..llama.mixin_llama import LlamaAttentionMixin, LlamaDecoderLayerMixin
from ..clip.mixin_clip import CLIPEncoderLayerAdaptersMixin, CLIPEncoderAdaptersMixin


class MllamaVisionAttentionAdaptersMixin(LlamaAttentionMixin):
    """Mixin for adding adapter support to MLLaMA's vision attention module."""


class MllamaTextCrossAttentionAdaptersMixin(LlamaAttentionMixin):
    """Mixin for adding adapter support to MLLaMA's cross-attention module."""


class MllamaTextSelfAttentionAdaptersMixin(LlamaAttentionMixin):
    """Mixin for adding adapter support to MLLaMA's self-attention module."""


class MllamaVisionEncoderLayerAdaptersMixin(CLIPEncoderLayerAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's vision encoder layers."""


class MllamaSelfAttentionDecoderLayerAdaptersMixin(LlamaDecoderLayerMixin):
    """Mixin for adding adapter support to MLLaMA's self-attention decoder layers."""


class MllamaCrossAttentionDecoderLayerAdaptersMixin(LlamaDecoderLayerMixin):
    """Mixin for adding adapter support to MLLaMA's cross-attention decoder layers."""


class MllamaVisionEncoderAdaptersMixin(CLIPEncoderAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's vision encoder module. """


class MllamaVisionModelAdaptersMixin(ModelBaseAdaptersMixin):
    """Adds adapters to the a MllamaVisionModel class."""

    support_prompt_tuning = False

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        # no embeddings therefore no post embedding forward

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        # Vision model has two encoders:
        # 1. local transformer focusing on fine-grained, tile-level features
        for i, layer in enumerate(self.transformer.layers):
            yield i, layer
        # 2. global transformer operating on output of the local transformer, integrating information across all tiles
        for i, layer in enumerate(self.global_transformer.layers, start=len(self.transformer.layers)):
            yield i, layer

    def post_embedding_forward(self, module, args, embedding_output):
        embedding_output = self.invertible_adapters_forward(embedding_output)
        # Prompt tuning not yet supported
        return embedding_output


class MllamaTextModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the a MllamaTextModel class."""

    support_prompt_tuning = False

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        # Register hook for post embedding forward
        self.embed_tokens.register_forward_hook(self.post_embedding_forward)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.layers):
            yield i, layer

    def post_embedding_forward(self, module, args, embedding_output):
        embedding_output = self.invertible_adapters_forward(embedding_output)
        # Prompt tuning not yet supported
        return embedding_output