from typing import Callable, Iterable, Tuple

import torch.nn as nn

from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import Linear as LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin


class DistilBertMultiHeadSelfAttentionMixin:
    """Adds adapters to the MultiHeadSelfAttention module of DistilBert."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.q_lin = LoRALinear.wrap(self.q_lin, "selfattn", model_config, adapters_config, attn_key="q")
        self.k_lin = LoRALinear.wrap(self.k_lin, "selfattn", model_config, adapters_config, attn_key="k")
        self.v_lin = LoRALinear.wrap(self.v_lin, "selfattn", model_config, adapters_config, attn_key="v")

        self.prefix_tuning = PrefixTuningLayer("self", model_config, adapters_config)


class DistilBertTransfomerBlockAdaptersMixin:
    """Adds adapters to the TransformerBlock module of DistilBert."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.ffn.lin1 = LoRALinear.wrap(self.ffn.lin1, "intermediate", model_config, adapters_config)
        self.ffn.lin2 = LoRALinear.wrap(self.ffn.lin2, "output", model_config, adapters_config)

        self.attention_adapters = BottleneckLayer("mh_adapter")
        self.output_adapters = BottleneckLayer("output_adapter")


class DistilBertTransformerAdaptersMixin:
    """Adds adapters to the Transformer module of DistilBert."""

    def forward(self, *args, **kwargs):
        if hasattr(self, "pre_forward_fn"):
            kwargs["x"] = self.pre_forward_fn(self, kwargs["x"])
        return super().forward(*args, **kwargs)


class DistilBertModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the DistilBert module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.transformer.layer):
            yield i, layer

    def _hook_fn(self, module, input):
        new_input = self.invertible_adapters_forward(input)
        return new_input

    def hook_after_embeddings(self, hook_fn: Callable):
        # PyTorch's built-in pre-forward hook does not pass the input ids.
        # Therefore, we need to use a custom hook.
        self.transformer.pre_forward_fn = hook_fn
