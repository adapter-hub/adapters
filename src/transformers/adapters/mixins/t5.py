from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import (
    EmbeddingAdaptersMixin,
    InvertibleAdaptersWrapperMixin,
    ModelAdaptersMixin,
    ModelWithHeadsAdaptersMixin,
)


class T5SelfAttentionLayerAdaptersMixin(AdapterLayer):
    def __init__(self):
        super().__init__("mh_adapter", None)


class T5CrossAttentionLayerAdaptersMixin(AdapterLayer):
    def __init__(self):
        super().__init__("cross_adapter", None)


class T5FFLayerAdaptersMixin(AdapterLayer):
    def __init__(self):
        super().__init__("output_adapter", None)


class T5ModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersWrapperMixin, ModelAdaptersMixin):
    """Adds adapters to the T5Model class."""

    invertible_adapters_base_name = "encoder"

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        global_i = 0
        if hasattr(self, "encoder"):
            global_i = len(self.encoder.block)
            for i, layer in enumerate(self.encoder.block):
                yield i, layer
        if hasattr(self, "decoder"):
            for i, layer in enumerate(self.decoder.block, start=global_i):
                yield i, layer


# EmbeddingAdaptersWrapperMixin not required here as base and heads model are identical
class T5ModelWithHeadsAdaptersMixin(ModelWithHeadsAdaptersMixin):
    pass
