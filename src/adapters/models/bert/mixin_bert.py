import logging
from typing import Iterable, Tuple

import torch.nn as nn

from ...composition import adjust_tensors_for_parallel_
from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin


logger = logging.getLogger(__name__)


class BertSelfAttentionAdaptersMixin:
    """Adds adapters to the BertSelfAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.query = LoRALinear.wrap(self.query, "selfattn", model_config, adapters_config, attn_key="q")
        self.key = LoRALinear.wrap(self.key, "selfattn", model_config, adapters_config, attn_key="k")
        self.value = LoRALinear.wrap(self.value, "selfattn", model_config, adapters_config, attn_key="v")

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )


# For backwards compatibility, BertSelfOutput inherits directly from BottleneckLayer
class BertSelfOutputAdaptersMixin(BottleneckLayer):
    """Adds adapters to the BertSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "mh_adapter"
        super().init_adapters(model_config, adapters_config)


# For backwards compatibility, BertOutput inherits directly from BottleneckLayer
class BertOutputAdaptersMixin(BottleneckLayer):
    """Adds adapters to the BertOutput module."""

    def __init__(self):
        super().__init__("output_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "output_adapter"
        super().init_adapters(model_config, adapters_config)


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.intermediate.dense = LoRALinear.wrap(
            self.intermediate.dense, "intermediate", model_config, adapters_config
        )
        self.output.dense = LoRALinear.wrap(self.output.dense, "output", model_config, adapters_config)

        # Set location keys for prefix tuning
        self.attention.self.location_key = "self"
        if hasattr(self, "add_cross_attention") and self.add_cross_attention:
            self.crossattention.self.location_key = "cross"


class BertModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        # Set hook for parallel composition
        for _, layer in self.iter_layers():
            self._set_layer_hook_for_parallel(layer)

        # Register hook for post embedding forward
        self.embeddings.register_forward_hook(self.post_embedding_forward)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, input):
            adjust_tensors_for_parallel_(input[0], input[1])
            return input

        layer.register_forward_pre_hook(hook)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer
