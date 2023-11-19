import unittest

from tests.methods.test_config_union import ConfigUnionAdapterTest
from transformers import ElectraConfig
from transformers.testing_utils import require_torch

from .composition.test_parallel import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
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
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class ElectraAdapterTestBase(AdapterTestBase):
    config_class = ElectraConfig
    config = make_config(
        ElectraConfig,
        # vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
    )
    tokenizer_name = "google/electra-base-generator"


@require_torch
class ElectraAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    PromptTuningTestMixin,
    UniPELTTestMixin,
    EmbeddingTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    ConfigUnionAdapterTest,
    ElectraAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class ElectraClassConversionTest(
    ModelClassConversionTestMixin,
    ElectraAdapterTestBase,
    unittest.TestCase,
):
    pass
