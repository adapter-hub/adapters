# coding=utf-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Whisper model."""

from typing import Callable, Optional, Tuple

import torch
from torch import nn

from transformers import EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoderLayer,
    WhisperEncoderLayer,
    eager_attention_forward,
)
from transformers.utils import logging

from ...composition import adjust_tensors_for_parallel, adjust_tensors_for_parallel_, match_attn_matrices_for_parallel
from .mixin_whisper import (
    WhisperAttentionAdaptersMixin,
    WhisperDecoderLayerAdaptersMixin,
    WhisperEncoderLayerAdaptersMixin,
)


logger = logging.get_logger(__name__)


class WhisperAttentionWithAdapters(WhisperAttentionAdaptersMixin, WhisperAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        is_cross_attention = key_value_states is not None

        bsz, tgt_len = hidden_states.shape[:-1]

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(query_states.shape[0], tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        is_updated = False
        if past_key_values is not None and isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True
                past_key_values = past_key_values.cross_attention_cache
            else:
                past_key_values = past_key_values.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values and is_updated:
            key_states = past_key_values.layers[self.layer_idx].keys
            value_states = past_key_values.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            key_states = key_states.view(key_states.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(value_states.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
            if past_key_values is not None:
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        # >>> START AH Changes <<<
        query_states, key_states, value_states = match_attn_matrices_for_parallel(
            query_states, key_states, value_states
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_states, attention_mask)

        key_states, value_states, attention_mask = self.prefix_tuning(
            key_states, value_states, hidden_states, attention_mask
        )
        (query_states,) = adjust_tensors_for_parallel(key_states, query_states)
        # >>> END AH Changes <<<

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
        )

        actual_bsz = attn_output.shape[0]
        attn_output = attn_output.reshape(actual_bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class WhisperEncoderLayerWithAdapters(WhisperEncoderLayerAdaptersMixin, WhisperEncoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # >>> START AH Changes <<<
        adjust_tensors_for_parallel_(hidden_states, attention_mask)
        # >>> END AH Changes <<<

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # >>> START AH Changes <<<
        hidden_states = self.attention_adapters(hidden_states, residual, None)
        # >>> END AH Changes <<<

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # >>> START AH Changes <<<
        hidden_states = self.output_adapters(hidden_states, residual, None)
        # >>> END AH Changes <<<

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, attn_weights


class WhisperDecoderLayerWithAdapters(WhisperDecoderLayerAdaptersMixin, WhisperDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # >>> START AH Changes <<<
        adjust_tensors_for_parallel_(hidden_states, attention_mask, encoder_attention_mask)
        # >>> END AH Changes <<<

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # >>> START AH Changes <<<
        hidden_states = self.attention_adapters(hidden_states, residual, None)
        # >>> END AH Changes <<<

        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            # >>> START AH Changes <<<
            hidden_states = self.cross_attention_adapters(hidden_states, residual, None)
            # >>> END AH Changes <<<

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # >>> START AH Changes <<<
        hidden_states = self.output_adapters(hidden_states, residual, None)
        # >>> END AH Changes <<<

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
