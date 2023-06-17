# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch XLM-RoBERTa model."""

import torch
import torch.utils.checkpoint

from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaOutput, XLMRobertaSelfOutput

from ...mixins.bert import BertOutputAdaptersMixin, BertSelfOutputAdaptersMixin


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput with Roberta->XLMRoberta
class XLMRobertaSelfOutputWithAdapters(BertSelfOutputAdaptersMixin, XLMRobertaSelfOutput):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter_layer_forward(hidden_states, input_tensor, self.LayerNorm)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaOutput with Roberta->XLMRoberta
class XLMRobertaOutputWithAdapters(BertOutputAdaptersMixin, XLMRobertaOutput):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter_layer_forward(hidden_states, input_tensor, self.LayerNorm)
        return hidden_states
