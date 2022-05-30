import unittest

from tests.models.deberta_v2.test_modeling_deberta_v2 import *
from transformers import DebertaV2AdapterModel
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
class DebertaV2AdapterModelTest(AdapterModelTesterMixin, DebertaV2ModelTest):
    all_model_classes = (
        DebertaV2AdapterModel,
    )


class DebertaV2AdapterTestBase(AdapterTestBase):
    config_class = DebertaV2Config
    config = make_config(
        DebertaV2Config,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
    )
    tokenizer_name = "microsoft/deberta-v3-base"


@require_torch
class DebertaV2AdapterTest(

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

    DebertaV2AdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class DebertaV2ClassConversionTest(
    ModelClassConversionTestMixin,
    DebertaV2AdapterTestBase,
    unittest.TestCase,
):
    pass
