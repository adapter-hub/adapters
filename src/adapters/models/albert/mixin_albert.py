from typing import Iterable, Tuple

import torch.nn as nn

from ...composition import adjust_tensors_for_parallel_
from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin


class AlbertAttentionAdaptersMixin:
    """Adds adapters to the AlbertAttention module of ALBERT."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.query = LoRALinear.wrap(self.query, "selfattn", model_config, adapters_config, attn_key="q")
        self.key = LoRALinear.wrap(self.key, "selfattn", model_config, adapters_config, attn_key="k")
        self.value = LoRALinear.wrap(self.value, "selfattn", model_config, adapters_config, attn_key="v")

        self.attention_adapters = BottleneckLayer("mh_adapter")

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )


class AlbertEncoderLayerAdaptersMixin:
    """Adds adapters to the AlbertLayer module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.ffn = LoRALinear.wrap(self.ffn, "intermediate", model_config, adapters_config)
        self.ffn_output = LoRALinear.wrap(self.ffn_output, "output", model_config, adapters_config)

        # Set location keys for prefix tuning
        self.location_key = "output_adapter"

        self.output_adapters = BottleneckLayer("output_adapter")

        self.attention.location_key = "self"


class AlbertModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the AlbertModel module."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        # Set hook for parallel composition
        for _, layer in self.iter_layers():
            self._set_layer_hook_for_parallel(layer)

        self.embeddings.register_forward_hook(self.post_embedding_forward)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, input):
            adjust_tensors_for_parallel_(input[0], input[1])
            return input

        layer.register_forward_pre_hook(hook)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        i = 0
        for albertLayerGroup in self.encoder.albert_layer_groups:
            for albertLayer in albertLayerGroup.albert_layers:
                yield i, albertLayer
                i += 1
