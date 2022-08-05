import unittest

from tests.models.electra.test_modeling_electra import *
from transformers import ElectraAdapterModel
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
class ElectraAdapterModelTest(AdapterModelTesterMixin, ElectraModelTest):
    all_model_classes = (
        ElectraAdapterModel,
    )


class ElectraAdapterTestBase(AdapterTestBase):
    config_class = ElectraConfig
    config = make_config(
        ElectraConfig,
        dim=32,
        n_layers=4,
        n_heads=4,
        hidden_dim=37,
    )
    tokenizer_name = "google/electra-small-discriminator"


@require_torch
class ElectraAdapterTest(
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
