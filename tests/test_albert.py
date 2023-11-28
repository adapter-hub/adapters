import unittest
from math import ceil

from transformers import AlbertConfig
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


class AlbertAdapterTestBase(AdapterTestBase):
    config_class = AlbertConfig
    config = make_config(
        AlbertConfig,
        embedding_size=16,
        hidden_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        num_hidden_groups=2,
    )
    tokenizer_name = "albert-base-v2"
    leave_out_layers = [0]


@require_torch
class AlbertAdapterTest(
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
    AlbertAdapterTestBase,
    unittest.TestCase,
):
    def test_context_simple(self):
        expected_number_of_adapter_calls = ceil(self.config().num_hidden_layers / self.config().num_hidden_groups)
        super().test_context_simple(expected_number_of_adapter_calls=expected_number_of_adapter_calls)


@require_torch
class AlbertClassConversionTest(
    ModelClassConversionTestMixin,
    AlbertAdapterTestBase,
    unittest.TestCase,
):
    pass
