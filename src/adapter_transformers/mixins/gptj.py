from typing import Callable, Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..lora import Linear as LoRALinear
from ..model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin
from ..prefix_tuning import PrefixTuningShim


class GPTJAttentionAdaptersMixin:
    def init_adapters(self, config):
        self.location_key = "self"

        # Wrap layers for LoRA
        self.q_proj = LoRALinear.wrap(self.q_proj, "selfattn", config, attn_key="q")
        self.k_proj = LoRALinear.wrap(self.k_proj, "selfattn", config, attn_key="k")
        self.v_proj = LoRALinear.wrap(self.v_proj, "selfattn", config, attn_key="v")

        PrefixTuningShim.wrap(
            self, self.location_key + "_prefix" if self.location_key else None, config, past_kv_name="layer_past"
        )


class GPTJMLPAdaptersMixin:
    def init_adapters(self, config):
        # Wrap layers for LoRA
        self.fc_in = LoRALinear.wrap(self.fc_in, "intermediate", config, attn_key="q")
        self.fc_out = LoRALinear.wrap(self.fc_out, "output", config, attn_key="k")


class GPTJDecoderBlockAdaptersMixin:
    """Adds adapters to the TransformerBlock module of GPTJ."""

    def init_adapters(self, config):
        self.attention_adapters = AdapterLayer("mh_adapter")
        self.output_adapters = AdapterLayer("output_adapter")


class GPTJModelAdapterMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    def init_adapters(self, config):
        super().init_adapters(config)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.base_model.h):
            yield i, layer

    def hook_after_embeddings(self, hook_fn: Callable):
        return self.wte.register_forward_hook(hook_fn)
