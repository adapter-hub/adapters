import logging
from typing import Callable, Iterable, Tuple

import torch.nn as nn

from ...composition import adjust_tensors_for_parallel_
from ...layer import AdapterLayer
from ...lora import Linear as LoRALinear
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin
from ...prefix_tuning import PrefixTuningShim

logger = logging.getLogger(__name__)


class WavLMSelfAttentionAdaptersMixin:
    """Adds adapters to the WavLMSelfAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        # self.location_key = ""

        # fixme: original name -> q_proj, k_proj, v_proj
        # self.query = LoRALinear.wrap(self.query, "selfattn", model_config, adapters_config, attn_key="q")
        # self.key = LoRALinear.wrap(self.key, "selfattn", model_config, adapters_config, attn_key="k")
        # self.value = LoRALinear.wrap(self.value, "selfattn", model_config, adapters_config, attn_key="v")

        self.q_proj = LoRALinear.wrap(self.q_proj, "selfattn", model_config, adapters_config, attn_key="q")
        self.k_proj = LoRALinear.wrap(self.k_proj, "selfattn", model_config, adapters_config, attn_key="k")
        self.v_proj = LoRALinear.wrap(self.v_proj, "selfattn", model_config, adapters_config, attn_key="v")

        # fixme: no need?
        self.prefix_tuning = PrefixTuningShim(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )


# For backwards compatibility, BertSelfOutput inherits directly from AdapterLayer
class WavLMSelfOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the WavlmSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "mh_adapter"
        super().init_adapters(model_config, adapters_config)


# For backwards compatibility, BertOutput inherits directly from AdapterLayer
class WavLMOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the WavlmOutput module."""

    def __init__(self):
        super().__init__("output_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "output_adapter"
        super().init_adapters(model_config, adapters_config)


class WavLMLayerAdaptersMixin:
    """Adds adapters to the WavLMLayer module."""
    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        # self.intermediate.dense = LoRALinear.wrap(
        #     self.intermediate.dense, "intermediate", model_config, adapters_config
        # )
        # self.output.dense = LoRALinear.wrap(self.output.dense, "output", model_config, adapters_config)
        # print("initializing adapters")
        # Fixme: key might not be "intermediate", "output"
        self.feed_forward.intermediate_dense = LoRALinear.wrap(
            self.feed_forward.intermediate_dense, "intermediate", model_config, adapters_config
        )
        self.feed_forward.output_dense = LoRALinear.wrap(self.feed_forward.output_dense,
                                                         "output", model_config, adapters_config)
        self.attention_adapters = AdapterLayer(location_key="mh_adapter")
        self.output_adapters = AdapterLayer(location_key="output_adapter")
        # Set location keys for prefix tuning
        self.attention.location_key = "."

        # fixme: wavlm encoder doesn't have cross attention attribute
        # if hasattr(self, "add_cross_attention") and self.add_cross_attention:
        #     self.crossattention.self.location_key = "cross"


class WavLMModelAdaptersMixin(ModelBaseAdaptersMixin):
    """Adds adapters to the WavLMModel module."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

    # todo:
        # # Set hook for parallel composition
        # for _, layer in self.iter_layers():
        #     self._set_layer_hook_for_parallel(layer)

    # def _set_layer_hook_for_parallel(self, layer: nn.Module):
    #     def hook(module, input):
    #         adjust_tensors_for_parallel_(input[0], input[1])
    #         return input
    #
    #     layer.register_forward_pre_hook(hook)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layers):
            yield i, layer

    # def hook_after_embeddings(self, hook_fn: Callable):
    #     return self.feature_projection.register_forward_hook(hook_fn)
