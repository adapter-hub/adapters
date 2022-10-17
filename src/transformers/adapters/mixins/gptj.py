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


class GPTJDecoderBlockAdaptersMixin:
    """Adds adapters to the TransformerBlock module of DistilBert."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class GPTJModelAdapterMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin):
    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.base_model.h):
            yield i, layer


class GPTJModelWithHeadsAdaptersMixin(EmbeddingAdaptersWrapperMixin, ModelWithHeadsAdaptersMixin):
    pass
