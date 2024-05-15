import unittest

from tests.methods.test_config_union import ConfigUnionAdapterTest
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForAudioClassification,
    WhisperForCausalLM,
    WhisperForConditionalGeneration,
)
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

    has_static_head = True




class WhisperForConditionalGenerationAdapterTestBase(WhisperAdapterTestBase):
    model_class = WhisperForConditionalGeneration



@require_torch
class WhisperForConditionalGenerationAdapterTest(
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
    WhisperForConditionalGenerationAdapterTestBase,
    unittest.TestCase,
):
    def test_adapter_fusion_save_with_head(self):
        # This test is not applicable to CLIP
        self.skipTest("Not applicable to static Whisper model.")


class WhisperForCausalLMAdapterTestBase(WhisperAdapterTestBase):
    model_class = WhisperForCausalLM


class WhisperForCausalLMAdapterTest(
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
    WhisperForCausalLMAdapterTestBase,
    unittest.TestCase,
):
    pass


class WhisperForAudioClassificationAdapterTestBase(WhisperAdapterTestBase):
    model_class = WhisperForAudioClassification


class WhisperForAudioClassificationAdapterTest(
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
    WhisperForAudioClassificationAdapterTestBase,
    unittest.TestCase,
):
    pass
