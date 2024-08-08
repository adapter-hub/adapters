# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Classes to support Encoder-Decoder architectures"""

from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel

from .mixin_encoder_decoder import EncoderDecoderModelAdaptersMixin


# Although this class is empty, we cannot add the mixin via the MODEL_MIXIN_MAPPING, as this would result in a circular import.
class EncoderDecoderModelWithAdapters(EncoderDecoderModelAdaptersMixin, EncoderDecoderModel):
    pass
