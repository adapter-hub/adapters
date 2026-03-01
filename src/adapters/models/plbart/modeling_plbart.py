# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch PLBART model."""
from typing import Callable, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.plbart.modeling_plbart import (
    PLBartAttention,
    PLBartDecoderLayer,
    PLBartEncoderLayer,
    eager_attention_forward,
)
from transformers.utils import logging

from ...composition import adjust_tensors_for_parallel, adjust_tensors_for_parallel_, match_attn_matrices_for_parallel
from .mixin_plbart import (
    PLBartAttentionAdaptersMixin,
    PLBartDecoderLayerAdaptersMixin,
    PLBartEncoderLayerAdaptersMixin,
)


logger = logging.get_logger(__name__)


class PLBartAttentionWithAdapters(PLBartAttentionAdaptersMixin, PLBartAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Loosen constraint on batch_size to allow parallel adapter composition
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(tensor.shape[0], seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        # >>> START AH Changes <<<
        # Replaced:
        # query_states = self.q_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self._shape(self.q_proj(hidden_states), -1, bsz)
        # >>> END AH Changes <<<

        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_states from cache
                    curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.layers[self.layer_idx].keys
            value_states = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            # >>> START AH Changes <<<
            # Replaced:
            # key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            # value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)
            # >>> END AH Changes <<<

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        # >>> START AH Changes <<<
        # Inserted (replaced nothing):
        query_states, key_states, value_states = match_attn_matrices_for_parallel(
            query_states, key_states, value_states
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_states, attention_mask)

        key_states, value_states, attention_mask = self.prefix_tuning(
            key_states, value_states, hidden_states, attention_mask
        )
        (query_states,) = adjust_tensors_for_parallel(key_states, query_states)
        bsz = query_states.size(0)
        # >>> END AH Changes <<<

        # >>> START AH Changes <<<
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_states.shape[2]]
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
            scaling=self.scaling,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class PLBartEncoderLayerWithAdapters(PLBartEncoderLayerAdaptersMixin, PLBartEncoderLayer):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # >>> START AH Changes <<<
        # Inserted (replaced nothing):
        adjust_tensors_for_parallel_(hidden_states, attention_mask)
        # >>> END AH Changes <<<

        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # >>> START AH Changes <<<
        # Replaced:
        # hidden_states = residual + hidden_states
        # hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.attention_adapters(hidden_states, residual, self.self_attn_layer_norm)
        # >>> END AH Changes <<<

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # >>> START AH Changes <<<
        # Replaced:
        # hidden_states = residual + hidden_states
        # hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.output_adapters(hidden_states, residual, self.final_layer_norm)
        # >>> END AH Changes <<<

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class PLBartDecoderLayerWithAdapters(PLBartDecoderLayerAdaptersMixin, PLBartDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
                cache in the correct position and to infer the complete sequence length.
        """
        adjust_tensors_for_parallel_(hidden_states, attention_mask, encoder_attention_mask)

        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_values,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # >>> START AH Changes <<<
        # Replaced:
        # hidden_states = residual + hidden_states
        # hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.attention_adapters(hidden_states, residual, self.self_attn_layer_norm)
        # >>> END AH Changes <<<

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights, past_key_values = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            # >>> START AH Changes <<<
            # Replaced:
            # hidden_states = residual + hidden_states
            # hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.cross_attention_adapters(hidden_states, residual, self.encoder_attn_layer_norm)
            # >>> END AH Changes <<<

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # >>> START AH Changes <<<
        # Replaced:
        # hidden_states = residual + hidden_states
        # hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.output_adapters(hidden_states, residual, self.final_layer_norm)
        # >>> END AH Changes <<<

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (past_key_values,)

        return outputs
