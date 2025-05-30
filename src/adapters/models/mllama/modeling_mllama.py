# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from adapters.composition import adjust_tensors_for_parallel, match_attn_matrices_for_parallel
from transformers.cache_utils import Cache
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaTextCrossAttention,
    MllamaTextCrossSdpaAttention,
    MllamaTextSelfAttention,
    MllamaTextSelfSdpaAttention,
    MllamaVisionAttention,
    MllamaVisionSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import logging

from .mixin_mllama import (
    MllamaCrossAttentionDecoderLayerAdaptersMixin,
    MllamaSelfAttentionDecoderLayerAdaptersMixin,
    MllamaTextCrossAttentionAdaptersMixin,
    MllamaTextSelfAttentionAdaptersMixin,
    MllamaVisionAttentionAdaptersMixin,
)


logger = logging.get_logger(__name__)


class MllamaVisionAttentionWithAdapters(MllamaVisionAttentionAdaptersMixin, MllamaVisionAttention):

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
    ) -> torch.Tensor:
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # >>> START AH Changes <<<
        query, key, value = match_attn_matrices_for_parallel(query, key, value)
        (attention_mask,) = adjust_tensors_for_parallel(query, attention_mask)
        # >>> END AH Changes <<<

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return output, attn_weights


class MllamaVisionSdpaAttentionWithAdapters(MllamaVisionAttentionAdaptersMixin, MllamaVisionSdpaAttention):

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
    ) -> torch.Tensor:
        if output_attentions:
            logger.warning_once(
                "MllamaModel is using MllamaVisionSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # >>> START AH Changes <<<
        query, key, value = match_attn_matrices_for_parallel(query, key, value)
        (attention_mask,) = adjust_tensors_for_parallel(query, attention_mask)
        # >>> END AH Changes <<<

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        output = self.o_proj(attn_output)

        return output, None


class MllamaTextCrossAttentionWithAdapters(MllamaTextCrossAttentionAdaptersMixin, MllamaTextCrossAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            key_states = self.k_norm(key_states)
            if past_key_value is not None:
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
        elif cache_position[0] != 0:
            key_states, value_states = (
                past_key_value.key_cache[self.layer_idx],
                past_key_value.value_cache[self.layer_idx],
            )
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )

        # >>> START AH Changes <<<
        query_states, key_states, value_states = match_attn_matrices_for_parallel(
            query_states, key_states, value_states
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_states, attention_mask)
        # >>> END AH Changes <<<

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MllamaTextCrossSdpaAttentionWithAdapters(MllamaTextCrossAttentionAdaptersMixin, MllamaTextCrossSdpaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MllamaModel is using MllamaTextCrossSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                cross_attention_states=cross_attention_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            if past_key_value is not None:
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
        elif cache_position[0] != 0:
            key_states, value_states = (
                past_key_value.key_cache[self.layer_idx],
                past_key_value.value_cache[self.layer_idx],
            )
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )

        # >>> START AH Changes <<<
        query_states, key_states, value_states = match_attn_matrices_for_parallel(
            query_states, key_states, value_states
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_states, attention_mask)
        # >>> END AH Changes <<<

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states = self.k_norm(key_states)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if attention_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class MllamaTextSelfAttentionWithAdapters(MllamaTextSelfAttentionAdaptersMixin, MllamaTextSelfAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # >>> START AH Changes <<<
        query_states, key_states, value_states = match_attn_matrices_for_parallel(
            query_states, key_states, value_states
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_states, attention_mask)
        # >>> END AH Changes <<<

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # >>> START AH Changes <<<
        key_states, value_states, attention_mask = self.prefix_tuning(
            key_states, value_states, hidden_states, attention_mask
        )
        (query_states,) = adjust_tensors_for_parallel(key_states, query_states)
        bsz = key_states.shape[0]
        # >>> END AH Changes <<<

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MllamaTextSelfSdpaAttentionWithAdapters(MllamaTextSelfAttentionAdaptersMixin, MllamaTextSelfSdpaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MllamaModel is using MllamaTextSelfSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # >>> START AH Changes <<<
        query_states, key_states, value_states = match_attn_matrices_for_parallel(
            query_states, key_states, value_states
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_states, attention_mask)
        # >>> END AH Changes <<<

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # >>> START AH Changes <<<
        key_states, value_states, attention_mask = self.prefix_tuning(
            key_states, value_states, hidden_states, attention_mask
        )
        (query_states,) = adjust_tensors_for_parallel(key_states, query_states)
        bsz = key_states.shape[0]
        # >>> END AH Changes <<<

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


class MllamaSelfAttentionDecoderLayerWithAdapters(
    MllamaSelfAttentionDecoderLayerAdaptersMixin, MllamaSelfAttentionDecoderLayer
):

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        # >>> START AH Changes <<<
        hidden_states = self.attention_adapters(hidden_states, residual, None)
        # >>> END AH Changes <<<

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # >>> START AH Changes <<<
        hidden_states = self.output_adapters(hidden_states, residual, None)
        # >>> END AH Changes <<<

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MllamaCrossAttentionDecoderLayerWithAdapters(
    MllamaCrossAttentionDecoderLayer, MllamaCrossAttentionDecoderLayerAdaptersMixin
):

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights, past_key_value = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        # >>> START AH Changes <<<
        hidden_states = self.attention_adapters(hidden_states, residual, None)
        # >>> END AH Changes <<<
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # >>> START AH Changes <<<
        hidden_states = self.output_adapters(hidden_states, residual, None)
        # >>> END AH Changes <<<

        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs
