from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import (
    EmbeddingAdaptersMixin,
    InvertibleAdaptersMixin,
    InvertibleAdaptersWrapperMixin,
    ModelBaseAdaptersMixin,
    ModelWithHeadsAdaptersMixin,
)


class T5AttentionAdaptersMixin:
    """Adds adapters to the T5Attention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.q = LoRALinear.wrap(self.q, "selfattn", model_config, adapters_config, attn_key="q", bias=False)
        self.k = LoRALinear.wrap(self.k, "selfattn", model_config, adapters_config, attn_key="k", bias=False)
        self.v = LoRALinear.wrap(self.v, "selfattn", model_config, adapters_config, attn_key="v", bias=False)

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )


class T5SelfAttentionLayerAdaptersMixin(BottleneckLayer):
    def __init__(self):
        super().__init__("mh_adapter", None)

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "mh_adapter"
        super().init_adapters(model_config, adapters_config)


class T5CrossAttentionLayerAdaptersMixin(BottleneckLayer):
    def __init__(self):
        super().__init__("cross_adapter", None)

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "cross_adapter"
        self.EncDecAttention.location_key = "cross"
        super().init_adapters(model_config, adapters_config)


class T5FFLayerAdaptersMixin(BottleneckLayer):
    def __init__(self):
        super().__init__("output_adapter", None)

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "output_adapter"
        super().init_adapters(model_config, adapters_config)

        if hasattr(self.DenseReluDense, "wi_1"):
            self.DenseReluDense.wi_1 = LoRALinear.wrap(
                self.DenseReluDense.wi_1, "intermediate", model_config, adapters_config
            )
        else:
            self.DenseReluDense.wi = LoRALinear.wrap(
                self.DenseReluDense.wi, "intermediate", model_config, adapters_config
            )
        self.DenseReluDense.wo = LoRALinear.wrap(self.DenseReluDense.wo, "output", model_config, adapters_config)


class T5BlockAdaptersMixin:
    """Adds adapters to the T5Block module."""

    def init_adapters(self, model_config, adapters_config):
        location_key = "self" if self.is_decoder else "encoder"
        self.layer[0].SelfAttention.location_key = location_key
        self.layer[-1].location_key = location_key


class T5StackAdaptersMixin(InvertibleAdaptersMixin):
    """Adds adapters to the T5Stack module."""

    def init_adapters(self, model_config, adapters_config):
        if not self.is_decoder:
            InvertibleAdaptersMixin.init_adapters(self, self.config, adapters_config)

    def post_embedding_forward(self, embedding_output):
        embedding_output = self.invertible_adapters_forward(embedding_output)
        # Prompt tuning not yet supported
        return embedding_output


class T5ModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersWrapperMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the T5Model class."""

    invertible_adapters_base_name = "encoder"
    support_prompt_tuning = False

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        global_i = 0
        if hasattr(self, "encoder"):
            global_i = len(self.encoder.block)
            for i, layer in enumerate(self.encoder.block):
                yield i, layer
        if hasattr(self, "decoder"):
            for i, layer in enumerate(self.decoder.block, start=global_i):
                yield i, layer


# Stating "labels" and "input_ids" explicitly is required for training using Trainer class
class T5ForCondiditionalGenerationWithHeadsMixin(ModelWithHeadsAdaptersMixin, T5ModelAdaptersMixin):
    def forward(
        self,
        *args,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        return super().forward(*args, input_ids=input_ids, labels=labels, **kwargs)


# Stating "start_positions"/"end_positions" and "input_ids" explicitly is required for training using Trainer class
class T5ForQuestionAnsweringWithHeadsMixin(ModelWithHeadsAdaptersMixin, T5ModelAdaptersMixin):
    def forward(
        self,
        *args,
        input_ids: Optional[torch.LongTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        return super().forward(
            *args, input_ids=input_ids, start_positions=start_positions, end_positions=end_positions, **kwargs
        )
