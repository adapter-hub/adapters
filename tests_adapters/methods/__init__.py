# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The Adapter-Hub Team. All rights reserved.
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

from .base import create_twin_models
from .test_adapter_common import BottleneckAdapterTestMixin
from .test_compacter import CompacterTestMixin
from .test_ia3 import IA3TestMixin
from .test_lora import LoRATestMixin
from .test_prefix_tuning import PrefixTuningTestMixin
from .test_prompt_tuning import PromptTuningTestMixin
from .test_unipelt import UniPELTTestMixin
