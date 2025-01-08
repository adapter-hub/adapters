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
from ...utils import patch_forward


class MllamaBaseAttentionAdaptersMixin:
    """Base mixin class for adding adapter support to attention modules in MLLaMA.

    Implements common adapter functionality for all attention variants including:
    - LoRA adapters for query, key, and value projections
    - Additional Prefix tuning layer

    This base implementation ensures consistent adapter behavior across different
    attention mechanisms in the model.
    """

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.q_proj = LoRALinear.wrap(self.q_proj, "selfattn", model_config, adapters_config, attn_key="q")
        self.k_proj = LoRALinear.wrap(self.k_proj, "selfattn", model_config, adapters_config, attn_key="k")
        self.v_proj = LoRALinear.wrap(self.v_proj, "selfattn", model_config, adapters_config, attn_key="v")

        self.prefix_tuning = PrefixTuningLayer(
            "self_prefix", model_config, adapters_config, add_model_type_to_key=True
        )
        patch_forward(self)


class MllamaVisionAttentionAdaptersMixin(MllamaBaseAttentionAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's vision attention module."""


class MllamaTextCrossAttentionAdaptersMixin(MllamaBaseAttentionAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's cross-attention module."""


class MllamaTextSelfAttentionAdaptersMixin(MllamaBaseAttentionAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's self-attention module."""


class MllamaBaseLayerAdaptersMixin:
    """Base mixin class for adding adapter support to MLLaMA layer modules.

    Implements common layer-level adapter functionality including:
    - LoRA adapters for MLP layers (fc1/fc2)
    - Bottleneck adapters for attention and output
    - Forward pass patching for adapter integration
    """

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.mlp.fc1 = LoRALinear.wrap(self.mlp.fc1, "intermediate", model_config, adapters_config)
        self.mlp.fc2 = LoRALinear.wrap(self.mlp.fc2, "output", model_config, adapters_config)

        self.attention_adapters = BottleneckLayer("mh_adapter")
        self.output_adapters = BottleneckLayer("output_adapter")

        patch_forward(self)


class MllamaVisionEncoderLayerAdaptersMixin(MllamaBaseLayerAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's vision encoder layers."""


class MllamaSelfAttentionDecoderLayerAdaptersMixin(MllamaBaseLayerAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's self-attention decoder layers."""


class MllamaCrossAttentionDecoderLayerAdaptersMixin(MllamaBaseLayerAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's cross-attention decoder layers."""


class MllamaVisionEncoderAdaptersMixin:
    """Mixin for adding adapter support to MLLaMA's vision encoder module.

    Implements parallel composition support for vision encoder layers by:
    - Setting up hooks to adjust tensors during forward pass for parallel adapter processing
    """

    def init_adapters(self, model_config, adapters_config):
        # Set hook for parallel composition
        for layer in self.layers:
            self._set_layer_hook_for_parallel(layer)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, input):
            adjust_tensors_for_parallel_(input[0], input[1])
            return input

        layer.register_forward_pre_hook(hook)


class MllamaVisionModelAdaptersMixin(ModelBaseAdaptersMixin):
    """Adds adapters to the a MllamaVisionModel class."""

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
