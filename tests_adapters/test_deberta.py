import unittest

from tests.deberta.test_modeling_deberta import *
from transformers import DebertaAdapterModel
from transformers.testing_utils import require_torch

from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_common import AdapterModelTesterMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
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
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )


@require_torch
class DebertaAdapterTest(
    AdapterModelTesterMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
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
