import unittest

from tests_adapters.methods.test_config_union import ConfigUnionAdapterTest
from transformers import WavLMConfig
from transformers.testing_utils import require_torch

from .composition.test_parallel import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class WavLMAdapterTestBase(AdapterTestBase):
    config_class = WavLMConfig
    config = make_config(
        WavLMConfig,
        # hidden_size=32,
        # num_hidden_layers=5,
        # num_attention_heads=4,
        # intermediate_size=37,
        # hidden_act="gelu",
        # relative_attention=True,
        # pos_att_type="p2c|c2p",
    )
    feature_extractor_name = "microsoft/wavlm-base"


@require_torch
class WavLMAdapterTest(
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    EmbeddingTestMixin,
    ParallelTrainingMixin,
    ConfigUnionAdapterTest,
    WavLMAdapterTestBase,
    unittest.TestCase,
):
    pass

@require_torch
class DebertaClassConversionTest(
    ModelClassConversionTestMixin,
    WavLMAdapterTestBase,
    unittest.TestCase,
):
    pass
