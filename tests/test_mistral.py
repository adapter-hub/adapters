import unittest

from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.testing_utils import require_torch

from .composition.test_parallel import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    ReftTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class MistralAdapterTestBase(AdapterTestBase):
    config_class = MistralConfig
    config = make_config(
        MistralConfig,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=8,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        pad_token_id=0,
    )
    tokenizer_name = "HuggingFaceH4/zephyr-7b-beta"


@require_torch
class MistralAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    ReftTestMixin,
    UniPELTTestMixin,
    EmbeddingTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    MistralAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class MistralClassConversionTest(
    ModelClassConversionTestMixin,
    MistralAdapterTestBase,
    unittest.TestCase,
):
    pass
