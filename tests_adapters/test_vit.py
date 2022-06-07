import random
import unittest

from tests.models.vit.test_modeling_vit import *
from transformers import ViTAdapterModel
from transformers.testing_utils import require_torch

from .methods import BottleneckAdapterTestMixin, CompacterTestMixin, LoRATestMixin, PrefixTuningTestMixin
from .test_adapter import VisionAdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class ViTAdapterModelTest(AdapterModelTesterMixin, ViTModelTest):
    all_model_classes = (
        ViTAdapterModel,
    )


class ViTAdapterTestBase(VisionAdapterTestBase):
    config_class = ViTConfig
    config = make_config(
        ViTConfig,
        image_size=224,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    feature_extractor_name = 'google/vit-base-patch16-224-in21k'


@require_torch
class ViTAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    ViTAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class ViTClassConversionTest(
    ModelClassConversionTestMixin,
    ViTAdapterTestBase,
    unittest.TestCase,
):
    pass
