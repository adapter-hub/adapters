import unittest

from tests.methods.test_config_union import ConfigUnionAdapterTest
from transformers import GPTJConfig
from transformers.testing_utils import require_torch

from .composition.test_parallel import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
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
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class GPTJAdapterTestBase(AdapterTestBase):
    config_class = GPTJConfig
    config = make_config(
        GPTJConfig,
        n_embd=32,
        n_layer=4,
        n_head=4,
        rotary_dim=4,
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
    ConfigUnionAdapterTest,
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
