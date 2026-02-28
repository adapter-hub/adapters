# coding=utf-8
# Copyright 2023 Meta AI Team and the HuggingFace Inc. team.
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
"""PyTorch X-MOD model."""

import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.models.xmod.modeling_xmod import XmodOutput, XmodSelfAttention, XmodSelfOutput

from ...composition import adjust_tensors_for_parallel, match_attn_matrices_for_parallel
from ...utils import prefix_attention_mask
from ..bert.mixin_bert import BertOutputAdaptersMixin, BertSelfAttentionAdaptersMixin, BertSelfOutputAdaptersMixin


class XmodSelfAttentionWithAdapters(BertSelfAttentionAdaptersMixin, XmodSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        attention_mask = prefix_attention_mask(attention_mask)  # type: ignore

        batch_size, seq_length, _ = hidden_states.shape
        query_layer = self.query(hidden_states)
        query_layer = query_layer.view(
            query_layer.shape[0], -1, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

        is_updated = False
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_layer = self.key(current_states)
            key_layer = key_layer.view(
                key_layer.shape[0], -1, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)
            value_layer = self.value(current_states)
            value_layer = value_layer.view(
                value_layer.shape[0], -1, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)

            if past_key_values is not None:
                cache_position_arg = cache_position if not is_cross_attention else None
                key_layer, value_layer = curr_past_key_value.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position_arg}
                )
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        # >>> START AH Changes <<<
        query_layer, key_layer, value_layer = match_attn_matrices_for_parallel(query_layer, key_layer, value_layer)
        (attention_mask,) = adjust_tensors_for_parallel(query_layer, attention_mask)

        key_layer, value_layer, attention_mask = self.prefix_tuning(
            key_layer, value_layer, hidden_states, attention_mask
        )
        (query_layer,) = adjust_tensors_for_parallel(key_layer, query_layer)
        (value_layer,) = adjust_tensors_for_parallel(key_layer, value_layer)
        # >>> END AH Changes <<<

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if past_key_values is not None:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs


class XmodSelfOutputWithAdapters(BertSelfOutputAdaptersMixin, XmodSelfOutput):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.bottleneck_layer_forward(hidden_states, input_tensor, None)
        return hidden_states


class XmodOutputWithAdapters(BertOutputAdaptersMixin, XmodOutput):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.adapter_layer_norm is not None:
            layer_norm = self.adapter_layer_norm
        elif self.adapter_reuse_layer_norm:
            layer_norm = self.LayerNorm
        hidden_states = self.bottleneck_layer_forward(hidden_states, input_tensor, layer_norm)
        return hidden_states
