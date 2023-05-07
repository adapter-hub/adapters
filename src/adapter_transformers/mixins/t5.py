from typing import Callable, Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..lora import Linear as LoRALinear
from ..model_mixin import (
    EmbeddingAdaptersMixin,
    InvertibleAdaptersMixin,
    InvertibleAdaptersWrapperMixin,
    ModelBaseAdaptersMixin,
    ModelWithHeadsAdaptersMixin,
)
from ..prefix_tuning import PrefixTuningShim


class T5AttentionAdaptersMixin:
    """Adds adapters to the T5Attention module."""

    def init_adapters(self, config):
        # Wrap layers for LoRA
        self.q = LoRALinear.wrap(self.q, "selfattn", config, attn_key="q", bias=False)
        self.k = LoRALinear.wrap(self.k, "selfattn", config, attn_key="k", bias=False)
        self.v = LoRALinear.wrap(self.v, "selfattn", config, attn_key="v", bias=False)

        self.prefix_tuning = PrefixTuningShim(self.location_key + "_prefix" if self.location_key else None, config)


class T5SelfAttentionLayerAdaptersMixin(AdapterLayer):
    def __init__(self):
        super().__init__("mh_adapter", None)

    def init_adapters(self, config):
        self.location_key = "mh_adapter"
        self.SelfAttention.location_key = "self"
        super().init_adapters(config)


class T5CrossAttentionLayerAdaptersMixin(AdapterLayer):
    def __init__(self):
        super().__init__("cross_adapter", None)

    def init_adapters(self, config):
        self.location_key = "cross_adapter"
        self.EncDecAttention.location_key = "cross"
        super().init_adapters(config)


class T5FFLayerAdaptersMixin(AdapterLayer):
    def __init__(self):
        super().__init__("output_adapter", None)

    def init_adapters(self, config):
        self.location_key = "output_adapter"
        super().init_adapters(config)

        if hasattr(self.DenseReluDense, "wi_1"):
            self.DenseReluDense.wi_1 = LoRALinear.wrap(self.DenseReluDense.wi_1, "intermediate", config)
        else:
            self.DenseReluDense.wi = LoRALinear.wrap(self.DenseReluDense.wi, "intermediate", config)
        self.DenseReluDense.wo = LoRALinear.wrap(self.DenseReluDense.wo, "output", config)


class T5BlockAdaptersMixin:
    """Adds adapters to the T5Block module."""

    def init_adapters(self, config):
        location_key = "self" if self.is_decoder else "encoder"
        self.layer[-1].location_key = location_key


class T5StackAdaptersMixin(InvertibleAdaptersMixin):
    """Adds adapters to the T5Stack module."""

    def init_adapters(self, config):
        if not self.is_decoder:
            InvertibleAdaptersMixin.init_adapters(self, self.config)


class T5ModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersWrapperMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the T5Model class."""

    invertible_adapters_base_name = "encoder"

    def init_adapters(self, config, **kwargs):
        super().init_adapters(config, **kwargs)
        self.encoder.config.adapters = config.adapters
        self.decoder.config.adapters = config.adapters

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        global_i = 0
        if hasattr(self, "encoder"):
            global_i = len(self.encoder.block)
            for i, layer in enumerate(self.encoder.block):
                yield i, layer
        if hasattr(self, "decoder"):
            for i, layer in enumerate(self.decoder.block, start=global_i):
                yield i, layer


class T5ModelAdaptersWithHeadsMixin(ModelWithHeadsAdaptersMixin, T5ModelAdaptersMixin):
    pass
