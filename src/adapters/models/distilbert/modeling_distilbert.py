# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

"""
PyTorch DistilBERT model adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM) and in
part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""


import math
from typing import Optional, Tuple

import torch
from torch import nn

from transformers.models.distilbert.modeling_distilbert import (
    DistilBertFlashAttention2,
    DistilBertSdpaAttention,
    MultiHeadSelfAttention,
    TransformerBlock,
)
from transformers.utils import is_flash_attn_2_available, logging

from ...composition import adjust_tensors_for_parallel, adjust_tensors_for_parallel_, match_attn_matrices_for_parallel
from ...utils import prefix_attention_mask
from .mixin_distilbert import DistilBertMultiHeadSelfAttentionMixin, DistilBertTransfomerBlockAdaptersMixin


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


class MultiHeadSelfAttentionWithAdapters(DistilBertMultiHeadSelfAttentionMixin, MultiHeadSelfAttention):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            # keep first dim due to parallel composition
            return x.view(x.shape[0], -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        # >>> START AH Changes <<<
        q, k, v = match_attn_matrices_for_parallel(q, k, v)
        (mask,) = adjust_tensors_for_parallel(q, mask)

        k, v, mask = self.prefix_tuning(k, v, value, mask, invert_mask=False)
        bs = k.size(0)  # reset for Parallel block
        (q,) = adjust_tensors_for_parallel(k, q)
        # >>> END AH Changes <<<

        mask_reshp = (bs, 1, 1, k.size(2))

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class DistilBertSdpaAttentionWithAdapters(DistilBertMultiHeadSelfAttentionMixin, DistilBertSdpaAttention):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        if output_attentions or head_mask is not None:
            logger.warning_once(
                "DistilBertSdpaAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support"
                " `output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but specifying"
                " the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be"
                ' removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                query,
                key,
                value,
                mask,
                head_mask,
                output_attentions,
            )

        batch_size, _, _ = query.size()
        dim_per_head = self.dim // self.n_heads

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            # keep first dim due to parallel composition
            return x.view(x.shape[0], -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        # >>> START AH Changes <<<
        q, k, v = match_attn_matrices_for_parallel(q, k, v)
        (mask,) = adjust_tensors_for_parallel(q, mask)

        k, v, mask = self.prefix_tuning(k, v, value, mask, invert_mask=False)
        (q,) = adjust_tensors_for_parallel(k, q)
        # >>> END AH Changes <<<

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and q.device.type == "cuda" and mask is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )

        attn_output = unshape(attn_output)
        attn_output = self.out_lin(attn_output)

        return (attn_output,)


class DistilBertFlashAttention2WithAdapters(DistilBertMultiHeadSelfAttentionMixin, DistilBertFlashAttention2):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        batch_size, q_length, dim = query.size()

        dim_per_head = self.dim // self.n_heads

        def reshape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(x.shape[0], -1, self.n_heads, dim_per_head)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query_states = reshape(self.q_lin(query))
        key_states = reshape(self.k_lin(key))
        value_states = reshape(self.v_lin(value))

        attn_dropout = self.config.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_lin.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_weights = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            mask,
            q_length,
            dropout=attn_dropout,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_weights_reshaped = attn_weights.reshape(batch_size, q_length, self.n_heads * dim_per_head)
        attn_output = self.out_lin(attn_weights_reshaped)

        if output_attentions:
            return (attn_output, attn_weights)
        else:
            return (attn_output,)


class TransformerBlockWithAdapters(DistilBertTransfomerBlockAdaptersMixin, TransformerBlock):
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        adjust_tensors_for_parallel_(x, attn_mask)
        attn_mask = prefix_attention_mask(attn_mask, dim=[2, 3], prefix_value=1)  # type: ignore

        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            if type(sa_output) is not tuple:
                raise TypeError(f"sa_output must be a tuple but it is {type(sa_output)} type")

            sa_output = sa_output[0]
        sa_output = self.attention_adapters(sa_output, x, self.sa_layer_norm)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_adapters(
            ffn_output, sa_output, self.output_layer_norm
        )  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output
