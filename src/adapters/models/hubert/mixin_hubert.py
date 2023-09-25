import logging
from typing import Callable, Iterable, Tuple

import torch.nn as nn

from ...composition import adjust_tensors_for_parallel_
from ...layer import AdapterLayer
from ...lora import Linear as LoRALinear
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin
from ...prefix_tuning import PrefixTuningShim

logger = logging.getLogger(__name__)


class HubertSelfAttentionAdaptersMixin:
    """Adds adapters to the HubertSelfAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.q_proj = LoRALinear.wrap(self.q_proj, "selfattn", model_config, adapters_config, attn_key="q")
        self.k_proj = LoRALinear.wrap(self.k_proj, "selfattn", model_config, adapters_config, attn_key="k")
        self.v_proj = LoRALinear.wrap(self.v_proj, "selfattn", model_config, adapters_config, attn_key="v")

        # fixme: no need?
        self.prefix_tuning = PrefixTuningShim(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )


class HubertLayerAdaptersMixin:
    """Adds adapters to the HubertLayer module."""
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
        self.attention.location_key = "attention"

        # fixme: Didn't work
        # if hasattr(self, "add_cross_attention") and self.add_cross_attention:
        #     self.crossattention.self.location_key = "cross"


class HubertModelAdaptersMixin(ModelBaseAdaptersMixin):
    """Adds adapters to the HubertModel module."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

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
