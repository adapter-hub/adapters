from typing import Callable, Iterable, Tuple

import torch.nn as nn

from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import Linear as LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin


class LlamaAttentionMixin:
    def init_adapters(self, model_config, adapters_config):
        self.q_proj = LoRALinear.wrap(self.q_proj, "selfattn", model_config, adapters_config, attn_key="q")
        self.k_proj = LoRALinear.wrap(self.k_proj, "selfattn", model_config, adapters_config, attn_key="k")
        self.v_proj = LoRALinear.wrap(self.v_proj, "selfattn", model_config, adapters_config, attn_key="v")

        self.prefix_tuning = PrefixTuningLayer("self_prefix", model_config, adapters_config)


class LlamaDecoderLayerMixin:
    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.mlp.down_proj = LoRALinear.wrap(self.mlp.down_proj, "intermediate", model_config, adapters_config)
        self.mlp.up_proj = LoRALinear.wrap(self.mlp.up_proj, "output", model_config, adapters_config)

        self.attention_adapters = BottleneckLayer("mh_adapter")
        self.output_adapters = BottleneckLayer("output_adapter")


class LlamaModelAdapterMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.layers):
            yield i, layer

    def hook_after_embeddings(self, hook_fn: Callable):
        return self.embed_tokens.register_forward_hook(hook_fn)
