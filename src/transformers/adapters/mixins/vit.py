from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import ModelAdaptersMixin


class ViTOutputAdaptersMixin:
    """Adds adapters to the ViTOutput module."""

    def _init_adapter_modules(self):
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.output_adapters._init_adapter_modules()


# Unlike BERT, self attention adapters are added to Layer module in ViT
class ViTLayerAdaptersMixin:
    """Adds adapters to the ViTSelfOutput module."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.attention_adapters._init_adapter_modules()


class ViTModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the ViTModel class."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer
