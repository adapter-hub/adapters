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
""" PyTorch WavLM model."""


import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from adapters.composition import adjust_tensors_for_parallel
from transformers.models.wavlm.modeling_wavlm import WavLMEncoderLayerStableLayerNorm, WavLMAttention, WavLMFeedForward

from .mixin_wavlm import WavLMLayerAdaptersMixin, WavLMOutputAdaptersMixin, WavLMSelfAttentionAdaptersMixin

Tensor = torch.Tensor
class WavLMSelfAttentionWithAdapters(WavLMSelfAttentionAdaptersMixin, WavLMAttention):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_bias: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            index=0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Attention layer with relative attention"""
        bsz, tgt_len, _ = hidden_states.size()

        # first pass of attention layer creates position bias
        if position_bias is None:
            position_bias = self.compute_bias(tgt_len, tgt_len)
            position_bias = (
                position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, tgt_len)
            )

        # Compute relative position bias:
        # 1) get reshape hidden_states
        gated_hidden_states = hidden_states.view(hidden_states.shape[:-1] + (self.num_heads, -1))
        gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)

        # 2) project hidden states
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        relative_position_proj = relative_position_proj.view(gated_hidden_states.shape[:-1] + (2, 4)).sum(-1)

        # 3) compute gate for position bias from projected hidden states
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0

        # 4) apply gate to position bias to compute gated position_bias
        gated_position_bias = gate_output.view(bsz * self.num_heads, -1, 1) * position_bias
        gated_position_bias = gated_position_bias.view((-1, tgt_len, tgt_len))

        attn_output, attn_weights = self.torch_multi_head_self_attention(
            hidden_states, attention_mask, gated_position_bias, output_attentions
        )

        return attn_output, attn_weights, position_bias


class WavLMEncoderLayerWithAdapters(WavLMLayerAdaptersMixin, WavLMEncoderLayerStableLayerNorm):
    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        attn_residual = hidden_states

        # WavLM SelfAttention
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        attn_residual = hidden_states
        hidden_states = self.attention_adapters.adapter_layer_forward(hidden_states, attn_residual, None)

        # WavLMIntermediate
        hidden_states = self.feed_forward.intermediate_dense(hidden_states)
        hidden_states = self.feed_forward.intermediate_act_fn(hidden_states)
        hidden_states = self.feed_forward.intermediate_dropout(hidden_states)

        # WavLMOutput
        hidden_states = self.feed_forward.output_dense(hidden_states)
        hidden_states = self.feed_forward.output_dropout(hidden_states)
        hidden_states = self.output_adapters.adapter_layer_forward(hidden_states, attn_residual, self.final_layer_norm)

        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs