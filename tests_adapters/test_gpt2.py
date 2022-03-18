import unittest

from tests.test_modeling_gpt2 import *
from transformers import GPT2AdapterModel
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
from .test_adapter_backward_compability import CompabilityTestMixin


@require_torch
class GPT2AdapterModelTest(AdapterModelTesterMixin, GPT2ModelTest):
    all_model_classes = (
        GPT2AdapterModel,
    )


class GPT2AdapterTestBase(AdapterTestBase):
    config_class = GPT2Config
    config = make_config(
        GPT2Config,
        n_embd=32,
        n_layer=4,
        n_head=4,
        # set pad token to eos token
        pad_token_id=50256,
    )
    tokenizer_name = "gpt2"


@require_torch
class GPT2AdapterTest(
    AdapterModelTestMixin,
    EmbeddingTestMixin,
    CompabilityTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    AdapterTrainingTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    GPT2AdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class GPT2ClassConversionTest(
    ModelClassConversionTestMixin,
    GPT2AdapterTestBase,
    unittest.TestCase,
):
    pass
