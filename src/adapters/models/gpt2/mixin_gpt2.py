from typing import Iterable, Tuple

import torch.nn as nn

from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import LoRALinear, LoRAMergedLinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin
from ...utils import patch_forward


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

        patch_forward(self)


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

        patch_forward(self)


class GPT2ModelAdapterMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    support_prompt_tuning = False
    support_lora_delta_w_svd = False

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        # Register hook for post embedding forward
        self.drop.register_forward_hook(self.post_embedding_forward)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.base_model.h):
            yield i, layer

    def post_embedding_forward(self, module, args, embedding_output):
        embedding_output = self.invertible_adapters_forward(embedding_output)
        # Prompt tuning not yet supported
        return embedding_output
