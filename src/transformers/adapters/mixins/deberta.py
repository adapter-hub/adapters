from typing import Iterable, Tuple

from torch import nn

from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


class DebertaSelfOutputAdaptersMixin:
    """Adds adapters to the BertSelfOutput module."""

    pass


class DebertaOutputAdaptersMixin:
    """Adds adapters to the module."""

    pass


class DebertaModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer
