from typing import Iterable, Tuple

import torch.nn as nn

from ...composition import adjust_tensors_for_parallel_
from ...methods.reft import ReftLayer, hook_fn
from ...model_mixin import (
    EmbeddingAdaptersMixin,
    EmbeddingAdaptersWrapperMixin,
    InvertibleAdaptersMixin,
    InvertibleAdaptersWrapperMixin,
    ModelBaseAdaptersMixin,
)
from ..clip.mixin_clip import CLIPAttentionAdaptersMixin, CLIPEncoderLayerAdaptersMixin
from ..llama.mixin_llama import LlamaDecoderLayerMixin


class MllamaVisionAttentionAdaptersMixin(CLIPAttentionAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's vision attention module."""


class MllamaTextCrossAttentionAdaptersMixin(CLIPAttentionAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's cross-attention module."""


class MllamaTextSelfAttentionAdaptersMixin(CLIPAttentionAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's self-attention module."""


class MllamaVisionEncoderLayerAdaptersMixin(CLIPEncoderLayerAdaptersMixin):
    """Mixin for adding adapter support to MLLaMA's vision encoder layers."""


class MllamaSelfAttentionDecoderLayerAdaptersMixin(LlamaDecoderLayerMixin):
    """Mixin for adding adapter support to MLLaMA's self-attention decoder layers."""


class MllamaCrossAttentionDecoderLayerAdaptersMixin(LlamaDecoderLayerMixin):
    """Mixin for adding adapter support to MLLaMA's cross-attention decoder layers."""


class MllamaVisionEncoderAdaptersMixin:
    """Mixin for adding adapter support to MLLaMA's vision encoder module."""

    def init_adapters(self, model_config, adapters_config):
        # Set hook for parallel composition
        for layer in self.layers:
            self._set_layer_hook_for_parallel(layer)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, args, kwargs):
            # Extract the hidden states from kwargs
            if "hidden_state" in kwargs:
                hidden_states = kwargs["hidden_state"]
                attention_mask = kwargs.get("attention_mask")
                if attention_mask is not None:
                    adjust_tensors_for_parallel_(hidden_states, attention_mask)
                    kwargs["hidden_state"] = hidden_states
                    kwargs["attention_mask"] = attention_mask
            return args, kwargs

        layer.register_forward_pre_hook(hook, with_kwargs=True)


class MllamaVisionModelAdaptersMixin:
    """Adds adapters to the a MllamaVisionModel class."""

    support_prompt_tuning = False

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        # Vision model has two encoders:
        # 1. local transformer focusing on fine-grained, tile-level features
        for i, layer in enumerate(self.transformer.layers):
            yield i, layer
        # 2. global transformer operating on output of the local transformer, integrating information across all tiles
        for i, layer in enumerate(self.global_transformer.layers, start=len(self.transformer.layers)):
            yield i, layer


class MllamaTextModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin):
    """Adds adapters to the a MllamaTextModel class."""

    support_prompt_tuning = False

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.layers):
            yield i, layer


class MllamaAdaptersMixin(EmbeddingAdaptersWrapperMixin, InvertibleAdaptersWrapperMixin, ModelBaseAdaptersMixin):
    """
    Adds adapters to the MLLaMA model, handling both vision and text components.
    """

    invertible_adapters_base_name = "language_model"
    support_prompt_tuning = False

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        layer_idx = 0

        # First iterate through vision model's local transformer layers
        for _, layer in self.vision_model.iter_layers():
            yield layer_idx, layer
            layer_idx += 1

        for _, layer in self.language_model.iter_layers():
            yield layer_idx, layer
            layer_idx += 1

    def _init_adapters_submodules(self, model_config, adapters_config):
        """Initialize adapters in vision and language models separately."""
        # transformers naming inconsistency: Add num_attention_heads to the model config for the vision model because it is by default represented by the parameter attention_head
        model_config.vision_config.num_attention_heads = model_config.vision_config.attention_heads

        # Initialize vision model adapters
        for module in self.vision_model.modules():
            if hasattr(module, "init_adapters"):
                module.init_adapters(model_config.vision_config, adapters_config)

        # Initialize language model adapters
        for module in self.language_model.modules():
            if hasattr(module, "init_adapters"):
                module.init_adapters(model_config.text_config, adapters_config)

        # Initialize ReFT for all layers if needed
        self._init_reft_layers(model_config, adapters_config)

    def _init_reft_layers(self, model_config, adapters_config):
        """Initialize ReFT layers for both vision and language components."""
        # Vision local transformer
        for _, layer in self.vision_model.iter_layers():
            if not hasattr(layer, "reft_layer"):
                layer.reft_layer = ReftLayer("output", model_config.vision_config, adapters_config)
                layer.register_forward_hook(hook_fn)

        # Language model layers
        for _, layer in self.language_model.iter_layers():
            if not hasattr(layer, "reft_layer"):
                layer.reft_layer = ReftLayer("output", model_config.text_config, adapters_config)
                layer.register_forward_hook(hook_fn)
