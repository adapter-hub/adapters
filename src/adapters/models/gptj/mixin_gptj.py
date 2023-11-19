from typing import Iterable, Tuple

import torch.nn as nn

from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin


class GPTJAttentionAdaptersMixin:
    def init_adapters(self, model_config, adapters_config):
        self.location_key = "self"

        # Wrap layers for LoRA
        self.q_proj = LoRALinear.wrap(self.q_proj, "selfattn", model_config, adapters_config, attn_key="q")
        self.k_proj = LoRALinear.wrap(self.k_proj, "selfattn", model_config, adapters_config, attn_key="k")
        self.v_proj = LoRALinear.wrap(self.v_proj, "selfattn", model_config, adapters_config, attn_key="v")

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )


class GPTJMLPAdaptersMixin:
    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.fc_in = LoRALinear.wrap(self.fc_in, "intermediate", model_config, adapters_config)
        self.fc_out = LoRALinear.wrap(self.fc_out, "output", model_config, adapters_config)


class GPTJDecoderBlockAdaptersMixin:
    """Adds adapters to the TransformerBlock module of GPTJ."""

    def init_adapters(self, model_config, adapters_config):
        self.attention_adapters = BottleneckLayer("mh_adapter")
        self.output_adapters = BottleneckLayer("output_adapter")


class GPTJModelAdapterMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    support_prompt_tuning = False

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
