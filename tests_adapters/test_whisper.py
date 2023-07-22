import unittest

from transformers import WhisperConfig
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    UniPELTTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .composition.test_parallel import ParallelAdapterInferenceTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class WhisperAdapterTestBase(AdapterTestBase):
    config_class = WhisperConfig
    config = make_config(
        WhisperConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        vocab_size=51865,
    )
    tokenizer_name = "openai/whisper-tiny"


@require_torch
class WhisperAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    WhisperAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class WhisperClassConversionTest(
    ModelClassConversionTestMixin,
    WhisperAdapterTestBase,
    unittest.TestCase,
):
    pass
