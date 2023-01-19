import unittest

from tests.models.clip.test_modeling_clip import *
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase, VisionAdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin

# TODO
# @require_torch
# class CLIPAdapterModelTest(CLIPModelTest):
#     all_model_classes = (
#         CLIPAdapterModel,
#     )
#     fx_compatible = False


class CLIPVisionAdapterTestBase(VisionAdapterTestBase):
    model_class = CLIPVisionModel
    config_class = CLIPVisionConfig
    config = make_config(
        CLIPVisionConfig,
        image_size=30,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    feature_extractor_name = 'openai/clip-vit-base-patch32'


@require_torch
class CLIPVisionAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    CLIPVisionAdapterTestBase,
    unittest.TestCase,
):
    pass


class CLIPTextAdapterTestBase(AdapterTestBase):
    model_class = CLIPTextModel
    config_class = CLIPTextConfig
    config = make_config(
        CLIPTextConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    tokenizer_name = 'openai/clip-vit-base-patch32'


@require_torch
class CLIPTextAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    CLIPTextAdapterTestBase,
    unittest.TestCase,
):
    pass


class CLIPAdapterTestBase(AdapterTestBase):
    config_class = CLIPConfig
    config = CLIPConfig.from_text_vision_configs(
        CLIPTextConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        ),
        CLIPVisionConfig(
            image_size=30,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )
    )
    tokenizer_name = 'openai/clip-vit-base-patch32'


@require_torch
class CLIPAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    CLIPAdapterTestBase,
    unittest.TestCase,
):
    pass
