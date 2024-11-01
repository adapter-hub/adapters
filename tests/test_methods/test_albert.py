from transformers import AlbertConfig

from .imports import *


class AlbertAdapterTestBase(TextAdapterTestBase):
    """Model configuration for testing methods on Albert."""

    config_class = AlbertConfig
    config = make_config(
        AlbertConfig,
        embedding_size=16,
        hidden_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        num_hidden_groups=2,
    )
    tokenizer_name = "albert-base-v2"
    leave_out_layers = [0]


@require_torch
class Core(
    AlbertAdapterTestBase,
    ModelClassConversionTestMixin,
    CompabilityTestMixin,
    AdapterFusionModelTestMixin,
    unittest.TestCase,
):
    def test_context_simple(self):
        expected_number_of_adapter_calls = ceil(self.config().num_hidden_layers / self.config().num_hidden_groups)
        super().test_context_simple(expected_number_of_adapter_calls=expected_number_of_adapter_calls)


@require_torch
class Composition(
    AlbertAdapterTestBase,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    unittest.TestCase,
):
    pass


@require_torch
class Heads(
    AlbertAdapterTestBase,
    PredictionHeadModelTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
class Embeddings(
    AlbertAdapterTestBase,
    EmbeddingTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
class ClassConversion(
    ModelClassConversionTestMixin,
    AlbertAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class PrefixTuning(
    AlbertAdapterTestBase,
    PrefixTuningTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
class PromptTuning(
    AlbertAdapterTestBase,
    PromptTuningTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
class ReFT(
    AlbertAdapterTestBase,
    ReftTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
class UniPELT(
    AlbertAdapterTestBase,
    UniPELTTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
class Compacter(
    AlbertAdapterTestBase,
    CompacterTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
class Bottleneck(
    AlbertAdapterTestBase,
    BottleneckAdapterTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
class IA3(
    AlbertAdapterTestBase,
    IA3TestMixin,
    unittest.TestCase,
):
    pass


@require_torch
class LoRA(
    AlbertAdapterTestBase,
    LoRATestMixin,
    unittest.TestCase,
):
    pass
