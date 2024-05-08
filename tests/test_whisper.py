import unittest

from tests.methods.test_config_union import ConfigUnionAdapterTest
from transformers import WhisperConfig, WhisperFeatureExtractor
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
from .test_adapter import AdapterTestBase, SpeechAdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class WhisperAdapterTestBase(SpeechAdapterTestBase):
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
    feature_extractor_name = "openai/whisper-small"
    processor_name = "openai/whisper-small"
    tokenizer_name = "openai/whisper-small"
    sampling_rate = 16000
    decoder_start_token_id = 50257
    # TODO: adjust in all testfiles
    batch_size = 2
    seq_length = 80
    log_mel_features_dim = 3000


@require_torch
class WhisperAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    EmbeddingTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    ConfigUnionAdapterTest,
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
