import unittest

from tests.models.distilbert.test_modeling_distilbert import *
from transformers import DistilBertAdapterModel
from transformers.testing_utils import require_torch

from .methods import BottleneckAdapterTestMixin, LoRATestMixin, CompacterTestMixin, PrefixTuningTestMixin
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class DistilBertAdapterModelTest(AdapterModelTesterMixin, DistilBertModelTest):
    all_model_classes = (
        DistilBertAdapterModel,
    )


class DistilBertAdapterTestBase(AdapterTestBase):
    config_class = DistilBertConfig
    config = make_config(
        DistilBertConfig,
        dim=32,
        n_layers=4,
        n_heads=4,
        hidden_dim=37,
    )
    tokenizer_name = "distilbert-base-uncased"


@require_torch
class DistilBertAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    EmbeddingTestMixin,
    CompabilityTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    DistilBertAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class DistilBertClassConversionTest(
    ModelClassConversionTestMixin,
    DistilBertAdapterTestBase,
    unittest.TestCase,
):
    pass
