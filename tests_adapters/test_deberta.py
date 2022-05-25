import unittest

from tests.models.deberta.test_modeling_deberta import *
from transformers import DebertaAdapterModel
from transformers.testing_utils import require_torch

from .methods import BottleneckAdapterTestMixin, CompacterTestMixin, LoRATestMixin, PrefixTuningTestMixin
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class DebertaAdapterModelTest(AdapterModelTesterMixin, DebertaModelTest):
    all_model_classes = (
        DebertaAdapterModel,
    )


class DebertaAdapterTestBase(AdapterTestBase):
    config_class = DebertaConfig
    config = make_config(
        DebertaConfig,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
    )
    tokenizer_name = "microsoft/deberta-base"


@require_torch
class DebertaAdapterTest(
    AdapterModelTesterMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,

    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    EmbeddingTestMixin,
    ParallelTrainingMixin,

    DebertaAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class DebertaClassConversionTest(
    ModelClassConversionTestMixin,
    DebertaAdapterTestBase,
    unittest.TestCase,
):
    pass
