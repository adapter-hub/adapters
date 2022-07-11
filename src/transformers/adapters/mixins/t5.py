from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import (
    EmbeddingAdaptersMixin,
    InvertibleAdaptersMixin,
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


class T5ModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the T5Model class."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        global_i = 0
        if hasattr(self, "encoder"):
            global_i = len(self.encoder.block)
            for i, layer in enumerate(self.encoder.block):
                yield i, layer
        if hasattr(self, "decoder"):
            for i, layer in enumerate(self.decoder.block, start=global_i):
                yield i, layer

    def _init_adapter_modules(self):
        if hasattr(self, "encoder"):
            # In T5, the invertible adapters are implemented by the encoder module.
            # Therefore, relay mixin calls to the encoder here.
            self.invertible_adapters = self.encoder.invertible_adapters
            self.add_invertible_adapter = self.encoder.add_invertible_adapter
            self.get_invertible_adapter = self.encoder.get_invertible_adapter
            self.enable_invertible_adapters = self.encoder.enable_invertible_adapters
            self.invertible_adapters_forward = self.encoder.invertible_adapters_forward
            self.delete_invertible_adapter = self.encoder.delete_invertible_adapter
        super()._init_adapter_modules()


# EmbeddingAdaptersWrapperMixin not required here as base and heads model are identical
class T5ModelWithHeadsAdaptersMixin(ModelWithHeadsAdaptersMixin):
    pass
