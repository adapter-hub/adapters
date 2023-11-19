import unittest

from tests.methods.test_config_union import ConfigUnionAdapterTest
from transformers import RobertaConfig
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


class RobertaAdapterTestBase(AdapterTestBase):
    config_class = RobertaConfig
    config = make_config(
        RobertaConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
        vocab_size=50265,
    )
    tokenizer_name = "roberta-base"


@require_torch
class RobertaAdapterTest(
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
    ConfigUnionAdapterTest,
    RobertaAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class RobertaClassConversionTest(
    ModelClassConversionTestMixin,
    RobertaAdapterTestBase,
    unittest.TestCase,
):
    pass
