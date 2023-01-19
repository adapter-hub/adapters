from typing import Iterable, Tuple

import torch.nn as nn


from ..layer import AdapterLayer
from ..model_mixin import (
    EmbeddingAdaptersMixin,
    EmbeddingAdaptersWrapperMixin,
    InvertibleAdaptersMixin,
    ModelAdaptersMixin,
    ModelWithHeadsAdaptersMixin,
)


class CLIPEncoderLayerAdaptersMixin:
    """Adds adapters to the CLIPEncoderLayer module of CLIP."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class CLIPTextModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the CLIPTextModel class."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.text_model.encoder.layers):
            yield i, layer


class CLIPVisionModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the a CLIPVisionModel class."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.vision_model.encoder.layers):
            yield i, layer


class CLIPModelAdaptersMixin(EmbeddingAdaptersWrapperMixin, ModelWithHeadsAdaptersMixin):
    """Adds adapters to the CLIPModel class."""

    def _init_adapter_modules(self):
        self.invertible_adapters = self.text_model.invertible_adapters
        self.add_invertible_adapter = self.text_model.add_invertible_adapter
        self.get_invertible_adapter = self.text_model.get_invertible_adapter
        self.enable_invertible_adapters = self.text_model.enable_invertible_adapters
        self.invertible_adapters_forward = self.text_model.invertible_adapters_forward
        super()._init_adapter_modules()

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.text_model.encoder.layers):
            yield i, layer
        for i, layer in enumerate(self.vision_model.encoder.layers):
            yield i, layer
