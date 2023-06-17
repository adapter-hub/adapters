# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BERT model specific for generation."""

import torch
import torch.utils.checkpoint

from transformers.models.bert_generation.modeling_bert_generation import BertGenerationOutput, BertGenerationSelfOutput

from ...mixins.bert import BertOutputAdaptersMixin, BertSelfOutputAdaptersMixin


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->BertGeneration
class BertGenerationSelfOutputWithAdapters(BertSelfOutputAdaptersMixin, BertGenerationSelfOutput):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter_layer_forward(hidden_states, input_tensor, self.LayerNorm)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->BertGeneration
class BertGenerationOutputWithAdapters(BertOutputAdaptersMixin, BertGenerationOutput):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter_layer_forward(hidden_states, input_tensor, self.LayerNorm)
        return hidden_states
