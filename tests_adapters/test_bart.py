import unittest

from tests.test_modeling_bart import *
from transformers import BartAdapterModel
from transformers.testing_utils import require_torch

from .test_adapter import AdapterTestBase, make_config
from .test_adapter_common import AdapterModelTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_adapter_training import AdapterTrainingTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class BartAdapterModelTest(AdapterModelTesterMixin, BartModelTest):
    all_model_classes = (
        BartAdapterModel,
    )


class BartAdapterTestBase(AdapterTestBase):
    config_class = BartConfig
    config = make_config(
        BartConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
    )
    tokenizer_name = "facebook/bart-base"


@require_torch
class BartAdapterTest(
    AdapterModelTestMixin,
    AdapterFusionModelTestMixin,
    EmbeddingTestMixin,
    PredictionHeadModelTestMixin,
    AdapterTrainingTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    BartAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class BartClassConversionTest(
    ModelClassConversionTestMixin,
    BartAdapterTestBase,
    unittest.TestCase,
):
    pass
