import unittest

from tests.methods.test_config_union import ConfigUnionAdapterTest
from transformers import BertConfig
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


class BertAdapterTestBase(AdapterTestBase):
    config_class = BertConfig
    config = make_config(
        BertConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    tokenizer_name = "bert-base-uncased"


@require_torch
class BertAdapterTest(
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
    BertAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class BertClassConversionTest(
    ModelClassConversionTestMixin,
    BertAdapterTestBase,
    unittest.TestCase,
):
    pass
