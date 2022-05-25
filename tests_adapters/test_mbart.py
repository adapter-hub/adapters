import unittest

from tests.models.mbart.test_modeling_mbart import *
from transformers import MBartAdapterModel
from transformers.testing_utils import require_torch

from .methods import BottleneckAdapterTestMixin, LoRATestMixin, CompacterTestMixin, PrefixTuningTestMixin
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_composition import ParallelAdapterInferenceTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class MBartAdapterModelTest(AdapterModelTesterMixin, MBartModelTest):
    all_model_classes = (
        MBartAdapterModel,
    )


class MBartAdapterTestBase(AdapterTestBase):
    config_class = MBartConfig
    config = make_config(
        MBartConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        vocab_size=250027,
    )
    tokenizer_name = "facebook/mbart-large-cc25"


@require_torch
class MBartAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    MBartAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class MBartClassConversionTest(
    ModelClassConversionTestMixin,
    MBartAdapterTestBase,
    unittest.TestCase,
):
    pass
