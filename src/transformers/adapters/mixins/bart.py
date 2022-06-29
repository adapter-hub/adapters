from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin


class BartEncoderLayerAdaptersMixin:
    """Adds adapters to the BartEncoderLayer module of BART."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class BartDecoderLayerAdaptersMixin(BartEncoderLayerAdaptersMixin):
    """Adds adapters to the BartDecoderLayer module of BART."""

    def _init_adapter_modules(self):
        super()._init_adapter_modules()
        self.cross_attention_adapters = AdapterLayer("cross_adapter", self.config)
        self.cross_attention_adapters._init_adapter_modules()


class BartModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the BartModel class."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        if hasattr(self, "encoder"):
            for i, layer in enumerate(self.encoder.layers):
                yield i, layer
            for i, layer in enumerate(self.decoder.layers, start=len(self.encoder.layers)):
                yield i, layer
        else:
            for i, layer in enumerate(self.decoder.layers):
                yield i, layer

    def _init_adapter_modules(self):
        if hasattr(self, "encoder"):
            # In BART, the invertible adapters are implemented by the encoder module.
            # Therefore, relay mixin calls to the encoder here.
            self.invertible_adapters = self.encoder.invertible_adapters
            self.add_invertible_adapter = self.encoder.add_invertible_adapter
            self.get_invertible_adapter = self.encoder.get_invertible_adapter
            self.enable_invertible_adapters = self.encoder.enable_invertible_adapters
            self.invertible_adapters_forward = self.encoder.invertible_adapters_forward
        super()._init_adapter_modules()
