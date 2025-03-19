# coding=utf-8
# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch BEiT model."""


import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from adapters.composition import adjust_tensors_for_parallel, match_attn_matrices_for_parallel
from transformers.models.beit.modeling_beit import (
    BeitLayer,
    BeitRelativePositionBias,
    BeitSdpaSelfAttention,
    BeitSelfAttention,
)
from transformers.utils import logging

from .mixin_beit import BeitLayerAdaptersMixin, BeitSelfAttentionAdaptersMixin


logger = logging.get_logger(__name__)


class BeitSelfAttentionWithAdapters(BeitSelfAttentionAdaptersMixin, BeitSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
        interpolate_pos_encoding: bool = False,
        resolution: Optional[Tuple[int]] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        key_layer, value_layer, _ = self.prefix_tuning(key_layer, value_layer, hidden_states)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Add relative position bias if present.
        if self.relative_position_bias is not None:
            height, width = resolution
            window_size = (height // self.config.patch_size, width // self.config.patch_size)
            attention_scores = attention_scores + self.relative_position_bias(
                window_size, interpolate_pos_encoding, dim_size=hidden_states.shape[1]
            )

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class BeitSdpaSelfAttentionWithAdapters(BeitSelfAttentionAdaptersMixin, BeitSdpaSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
        interpolate_pos_encoding: bool = False,
        resolution: Optional[Tuple[int]] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if output_attentions or head_mask is not None:
            logger.warning_once(
                "`BeitSdpaSelfAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not "
                "support `output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, "
                "but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                head_mask=head_mask,
                output_attentions=output_attentions,
                relative_position_bias=relative_position_bias,
                interpolate_pos_encoding=interpolate_pos_encoding,
                resolution=resolution,
            )

        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # >>> START AH Changes <<<
        query_layer, key_layer, value_layer = match_attn_matrices_for_parallel(query_layer, key_layer, value_layer)

        key_layer, value_layer, _ = self.prefix_tuning(key_layer, value_layer, hidden_states)
        (query_layer,) = adjust_tensors_for_parallel(key_layer, query_layer)
        # >>> END AH Changes <<<

        attn_bias = None
        if self.relative_position_bias is not None:
            height, width = resolution
            window_size = (height // self.config.patch_size, width // self.config.patch_size)
            attn_bias = self.relative_position_bias(
                window_size, interpolate_pos_encoding, dim_size=hidden_states.shape[1]
            )

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            if attn_bias is None:
                attn_bias = relative_position_bias
            else:
                attn_bias += relative_position_bias

        scaling = 1 / math.sqrt(self.attention_head_size)
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attn_bias,
            dropout_p=self.config.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=scaling,
        )
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, None


class BeitLayerWithAdapters(BeitLayerAdaptersMixin, BeitLayer):
    """This corresponds to the Block class in the timm implementation."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
        interpolate_pos_encoding: bool = False,
        resolution: Optional[Tuple[int]] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in BEiT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
            interpolate_pos_encoding=interpolate_pos_encoding,
            resolution=resolution,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # apply lambda_1 if present
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = self.attention_adapters.bottleneck_layer_forward(
            self.drop_path(attention_output), hidden_states, None
        )

        # in BEiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = self.output_adapters.bottleneck_layer_forward(self.drop_path(layer_output), hidden_states, None)

        outputs = (layer_output,) + outputs

        return outputs
