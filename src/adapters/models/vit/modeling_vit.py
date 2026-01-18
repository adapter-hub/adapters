# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ViT model."""


from typing import Callable, Optional, Tuple, Union

import torch

from adapters.composition import adjust_tensors_for_parallel, match_attn_matrices_for_parallel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.vit.modeling_vit import ViTLayer, ViTOutput, ViTSelfAttention, eager_attention_forward
from transformers.utils import logging

from .mixin_vit import ViTLayerAdaptersMixin, ViTOutputAdaptersMixin, ViTSelfAttentionAdaptersMixin


logger = logging.get_logger(__name__)


# Import ViTAttention for wrapping
from transformers.models.vit.modeling_vit import ViTAttention


class ViTSelfAttentionWithAdapters(ViTSelfAttentionAdaptersMixin, ViTSelfAttention):
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        # Inline transpose operations (transpose_for_scores removed in transformers v4.52.x)
        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

        query_layer, key_layer, value_layer = match_attn_matrices_for_parallel(query_layer, key_layer, value_layer)

        key_layer, value_layer, _ = self.prefix_tuning(key_layer, value_layer, hidden_states)
        (query_layer,) = adjust_tensors_for_parallel(key_layer, query_layer)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # In transformers v4.52.x, ViTSelfAttention.forward() always returns (context_layer, attention_probs)
        # regardless of output_attentions parameter
        return context_layer, attention_probs


class ViTAttentionWithAdapters(ViTAttention):
    """
    ViT Attention wrapper that accepts output_attentions parameter for compatibility.
    In transformers v4.52.x, ViTAttention.forward() doesn't accept output_attentions,
    but ViTLayer.forward() still passes it, so we need to intercept and ignore it.
    ViTAttention.forward() returns a single tensor in v4.52.x.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,  # Accept but ignore this parameter
    ) -> torch.Tensor:
        # Call parent's forward which only accepts hidden_states and head_mask
        # Parent returns single tensor in v4.52.x
        return super().forward(hidden_states, head_mask)


class ViTOutputWithAdapters(ViTOutputAdaptersMixin, ViTOutput):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.output_adapters.bottleneck_layer_forward(hidden_states, input_tensor, None)

        return hidden_states


class ViTLayerWithAdapters(ViTLayerAdaptersMixin, ViTLayer):
    """This corresponds to the Block class in the timm implementation."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # In transformers v4.52.x, ViTLayer.forward() returns just a tensor, not a tuple
        # The attention() call handles its own output_attentions internally
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, head_mask, output_attentions=output_attentions)

        # attention_output is a single tensor in v4.52.x (from ViTAttention)
        # first residual connection
        hidden_states = self.attention_adapters.bottleneck_layer_forward(attention_output, hidden_states, None)

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output
