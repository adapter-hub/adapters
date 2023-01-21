from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import (
    EmbeddingAdaptersMixin,
    EmbeddingAdaptersWrapperMixin,
    InvertibleAdaptersWrapperMixin,
    ModelAdaptersMixin,
)


class CLIPEncoderLayerAdaptersMixin:
    """Adds adapters to the CLIPEncoderLayer module of CLIP."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class CLIPTextModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersWrapperMixin, ModelAdaptersMixin):
    """Adds adapters to the CLIPTextModel class."""

    invertible_adapters_base_name = "text_model"

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.text_model.encoder.layers):
            yield i, layer


class CLIPVisionModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the a CLIPVisionModel class."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.vision_model.encoder.layers):
            yield i, layer


class CLIPModelAdaptersMixin(EmbeddingAdaptersWrapperMixin, InvertibleAdaptersWrapperMixin, ModelAdaptersMixin):
    """Adds adapters to the CLIPModel class."""

    invertible_adapters_base_name = "text_model"

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.text_model.encoder.layers):
            yield i, layer
        for i, layer in enumerate(self.vision_model.encoder.layers, start=len(self.text_model.encoder.layers)):
            yield i, layer
