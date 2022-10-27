import logging
from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import ModelAdaptersMixin, ModelWithHeadsAdaptersMixin


logger = logging.getLogger(__name__)


class BeitLayerAdaptersMixin:
    """Adds adapters to the BeitLayer module."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.attention_adapters._init_adapter_modules()

        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.output_adapters._init_adapter_modules()


class BeitModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the BeitModel module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer


class BeitModelWithHeadsAdaptersMixin(ModelWithHeadsAdaptersMixin):
    pass
