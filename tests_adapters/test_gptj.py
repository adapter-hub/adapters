import unittest

from tests.models.gptj.test_modeling_gptj import *
from transformers import GPTJAdapterModel
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class GPTJAdapterModelTest(AdapterModelTesterMixin, GPTJModelTest):
    all_model_classes = (
        GPTJAdapterModel,
    )
    fx_compatible = False


class GPTJAdapterTestBase(AdapterTestBase):
    config_class = GPTJConfig
    config = make_config(
        GPTJConfig,
        n_embd=32,
        n_layer=4,
        n_head=4,
        # set pad token to eos token
        pad_token_id=50256,
        resid_pdrop=0.1,
    )
    tokenizer_name = "EleutherAI/gpt-j-6B"


@require_torch
class GPTJAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    UniPELTTestMixin,
    PrefixTuningTestMixin,
    EmbeddingTestMixin,
    CompabilityTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    GPTJAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class GPTJClassConversionTest(
    ModelClassConversionTestMixin,
    GPTJAdapterTestBase,
    unittest.TestCase,
):
    pass
