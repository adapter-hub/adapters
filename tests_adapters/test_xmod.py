import unittest

from transformers import XmodConfig
from transformers.testing_utils import require_torch

from .composition.test_parallel import ParallelAdapterInferenceTestMixin
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
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class XmodAdapterTestBase(AdapterTestBase):
    config_class = XmodConfig
    config = make_config(
        XmodConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
        vocab_size=250002,
        max_position_embeddings=512,
        default_language="en_XX",
    )
    tokenizer_name = "xlm-roberta-base"


@require_torch
class XmodAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    PromptTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    XmodAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class XmodClassConversionTest(
    ModelClassConversionTestMixin,
    XmodAdapterTestBase,
    unittest.TestCase,
):
    pass
