import random
import unittest

from tests.models.vit.test_modeling_vit import *
from transformers import ViTAdapterModel
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
class ViTAdapterModelTest(AdapterModelTesterMixin, ViTModelTest):
    all_model_classes = (
        ViTAdapterModel,
    )


class ViTAdapterTestBase(AdapterTestBase):
    config_class = ViTConfig
    config = make_config(
        ViTConfig,
        image_size=30,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    default_input_samples_shape = (3, 3, 30, 30)

    def get_input_samples(self, shape=None, config=None):
        shape = shape or self.default_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.random())
        pixel_values = torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()
        in_data = {"pixel_values": pixel_values}

        return in_data

    def add_head(self, model, name, **kwargs):
        model.add_image_classification_head(name, **kwargs)


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
