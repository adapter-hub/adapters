from typing import Iterable, Tuple

import torch.nn as nn

from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import ModelBaseAdaptersMixin


class BeitSelfAttentionAdaptersMixin:
    def init_adapters(self, model_config, adapters_config):
        self.location_key = "self"

        # Wrap layers for LoRA
        self.query = LoRALinear.wrap(self.query, "selfattn", model_config, adapters_config, attn_key="q")
        self.key = LoRALinear.wrap(self.key, "selfattn", model_config, adapters_config, attn_key="k")
        self.value = LoRALinear.wrap(self.value, "selfattn", model_config, adapters_config, attn_key="v")

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )


class BeitIntermediateAdaptersMixin:
    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.dense = LoRALinear.wrap(self.dense, "intermediate", model_config, adapters_config)


class BeitOutputAdaptersMixin:
    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.dense = LoRALinear.wrap(self.dense, "output", model_config, adapters_config)


class BeitLayerAdaptersMixin:
    """Adds adapters to the BeitLayer module."""

    def init_adapters(self, model_config, adapters_config):
        self.attention_adapters = BottleneckLayer("mh_adapter")
        self.output_adapters = BottleneckLayer("output_adapter")


class BeitModelAdaptersMixin(ModelBaseAdaptersMixin):
    """Adds adapters to the BeitModel module."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)
        self.embeddings.register_forward_hook(self.post_embedding_forward)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer

    def set_input_embeddings(self, value):
        self.embeddings.patch_embeddings = value

    def post_embedding_forward(self, module, args, outputs):
        embedding_output, tup = outputs
        return super().post_embedding_forward(module, args, embedding_output), tup
