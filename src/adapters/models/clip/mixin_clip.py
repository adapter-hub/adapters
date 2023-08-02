from typing import Callable, Iterable, Tuple

import torch.nn as nn

from ...composition import adjust_tensors_for_parallel_
from ...layer import AdapterLayer
from ...lora import Linear as LoRALinear
from ...model_mixin import (
    EmbeddingAdaptersMixin,
    EmbeddingAdaptersWrapperMixin,
    InvertibleAdaptersMixin,
    InvertibleAdaptersWrapperMixin,
    ModelBaseAdaptersMixin,
)
from ...prefix_tuning import PrefixTuningShim


class CLIPAttentionAdaptersMixin:
    """Adds adapters to the CLIPAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.q_proj = LoRALinear.wrap(self.q_proj, "selfattn", model_config, adapters_config, attn_key="q")
        self.k_proj = LoRALinear.wrap(self.k_proj, "selfattn", model_config, adapters_config, attn_key="k")
        self.v_proj = LoRALinear.wrap(self.v_proj, "selfattn", model_config, adapters_config, attn_key="v")

        self.prefix_tuning = PrefixTuningShim("self_prefix", model_config, adapters_config, add_model_type_to_key=True)


class CLIPEncoderLayerAdaptersMixin:
    """Adds adapters to the CLIPEncoderLayer module of CLIP."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.mlp.fc1 = LoRALinear.wrap(self.mlp.fc1, "intermediate", model_config, adapters_config)
        self.mlp.fc2 = LoRALinear.wrap(self.mlp.fc2, "output", model_config, adapters_config)

        self.attention_adapters = AdapterLayer("mh_adapter")
        self.output_adapters = AdapterLayer("output_adapter")


class CLIPEncoderAdaptersMixin:
    """Adds adapters to the CLIPEncoder module of CLIP."""

    def init_adapters(self, model_config, adapters_config):
        # Set hook for parallel composition
        for layer in self.layers:
            self._set_layer_hook_for_parallel(layer)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, input):
            adjust_tensors_for_parallel_(input[0], input[1])
            return input

        layer.register_forward_pre_hook(hook)


class CLIPTextTransformerAdaptersMixin(InvertibleAdaptersMixin):
    """Adds adapters to the CLIPTextTransformer module of CLIP."""

    def hook_after_embeddings(self, hook_fn: Callable):
        return self.embeddings.register_forward_hook(hook_fn)


class CLIPTextModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersWrapperMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the CLIPTextModel class."""

    invertible_adapters_base_name = "text_model"

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.text_model.encoder.layers):
            yield i, layer


class CLIPVisionModelAdaptersMixin(ModelBaseAdaptersMixin):
    """Adds adapters to the a CLIPVisionModel class."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.vision_model.encoder.layers):
            yield i, layer


class CLIPModelAdaptersMixin(EmbeddingAdaptersWrapperMixin, InvertibleAdaptersWrapperMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the CLIPModel class."""

    invertible_adapters_base_name = "text_model"

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.text_model.encoder.layers):
            yield i, layer
        for i, layer in enumerate(self.vision_model.encoder.layers, start=len(self.text_model.encoder.layers)):
            yield i, layer

    def _init_adapters_submodules(self, model_config, adapters_config):
        # Initialize adapters in text and vision model separately
        for module in self.text_model.modules():
            if hasattr(module, "init_adapters"):
                module.init_adapters(model_config.text_config, adapters_config)
        for module in self.vision_model.modules():
            if hasattr(module, "init_adapters"):
                module.init_adapters(model_config.vision_config, adapters_config)
