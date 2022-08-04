from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


# For backwards compatibility, ElectraSelfOutput inherits directly from AdapterLayer
class ElectraSelfOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the ElectraSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter", None)


# For backwards compatibility, ElectraOutput inherits directly from AdapterLayer
class ElectraOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the ElectraOutput module."""

    def __init__(self):
        super().__init__("output_adapter", None)


class ElectraModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the Electra module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer
