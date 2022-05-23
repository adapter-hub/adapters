import unittest

from transformers import XLMRobertaConfig
from transformers.testing_utils import require_torch

from .test_adapter import AdapterTestBase, make_config
from .test_adapter_conversion import ModelClassConversionTestMixin


@require_torch
class XLMRobertaClassConversionTest(
    ModelClassConversionTestMixin,
    AdapterTestBase,
    unittest.TestCase,
):
    config_class = XLMRobertaConfig
    config = make_config(
        XLMRobertaConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
