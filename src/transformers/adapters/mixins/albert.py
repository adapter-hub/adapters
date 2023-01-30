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


class AlbertAttentionAdaptersMixin:
    """Adds adapters to the AlbertAttention module of ALBERT."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.attention_adapters._init_adapter_modules()


class AlbertModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the AlbertModel module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, albertLayerGroup in enumerate(self.encoder.albert_layer_groups):
            for albertLayer in albertLayerGroup.albert_layers:
                yield i, albertLayer

    def add_invertible_adapter(self, adapter_name: str):
        super().add_invertible_adapter(adapter_name, embedding_dim=self.config.embedding_size)


class AlbertEncoderLayerAdaptersMixin:
    """Adds adapters to the AlbertSelfOutput module."""

    def _init_adapter_modules(self):
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.output_adapters._init_adapter_modules()


class AlbertModelWithHeadsAdaptersMixin(EmbeddingAdaptersWrapperMixin, ModelWithHeadsAdaptersMixin):
    pass
