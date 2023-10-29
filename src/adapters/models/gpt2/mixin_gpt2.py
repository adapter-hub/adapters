from typing import Callable, Iterable, Tuple

import torch.nn as nn

from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import Linear as LoRALinear
from ...methods.lora import MergedLinear as LoRAMergedLinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin


class GPT2AttentionAdaptersMixin:
    """Adds adapters to the Attention module of GPT2."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        if not self.is_cross_attention:
            self.c_attn = LoRAMergedLinear.wrap(
                self.c_attn,
                "selfattn",
                model_config,
                adapters_config,
                fan_in_fan_out=True,
                no_init_bias=True,
            )

        location_key = "cross_prefix" if self.is_cross_attention else "self_prefix"
        self.prefix_tuning = PrefixTuningLayer(location_key, model_config, adapters_config)


class GPT2DecoderBlockAdaptersMixin:
    """Adds adapters to the TransformerBlock module of DistilBert."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.mlp.c_fc = LoRALinear.wrap(
            self.mlp.c_fc,
            "intermediate",
            model_config,
            adapters_config,
            fan_in_fan_out=True,
            no_init_bias=True,
        )
        self.mlp.c_proj = LoRALinear.wrap(
            self.mlp.c_proj,
            "output",
            model_config,
            adapters_config,
            fan_in_fan_out=True,
            no_init_bias=True,
        )

        self.attention_adapters = BottleneckLayer("mh_adapter")
        self.output_adapters = BottleneckLayer("output_adapter")


class GPT2ModelAdapterMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.base_model.h):
            yield i, layer

    def hook_after_embeddings(self, hook_fn: Callable):
        return self.drop.register_forward_hook(hook_fn)
