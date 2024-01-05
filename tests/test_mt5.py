import unittest

from transformers import MT5Config
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


@require_torch
class MT5AdapterTestBase(AdapterTestBase):
    config_class = MT5Config
    config = make_config(
        MT5Config,
        d_model=16,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=4,
        d_kv=16 // 4,
        tie_word_embeddings=False,
        decoder_start_token_id=0,
    )
    tokenizer_name = "google/mt5-base"


@require_torch
class MT5AdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    EmbeddingTestMixin,
    CompabilityTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    MT5AdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class MT5ClassConversionTest(
    ModelClassConversionTestMixin,
    MT5AdapterTestBase,
    unittest.TestCase,
):
    pass
