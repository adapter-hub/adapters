import unittest

from transformers import XLMRobertaConfig
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    PromptTuningTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin


class XLMRobertaAdapterTestBase(AdapterTestBase):
    config_class = XLMRobertaConfig
    config = make_config(
        XLMRobertaConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
        vocab_size=250002,
    )
    tokenizer_name = "xlm-roberta-base"


@require_torch
class XLMRobertaAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    PromptTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    XLMRobertaAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class XLMRobertaClassConversionTest(
    ModelClassConversionTestMixin,
    XLMRobertaAdapterTestBase,
    unittest.TestCase,
):
    pass
