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
@pytest.mark.core
class Core(
    AlbertAdapterTestBase,
    ModelClassConversionTestMixin,
    CompabilityTestMixin,
    AdapterFusionModelTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.composition
class Composition(
    AlbertAdapterTestBase,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.heads
class Heads(
    AlbertAdapterTestBase,
    PredictionHeadModelTestMixin,
    unittest.TestCase,
):
    def test_context_simple(self):
        expected_number_of_adapter_calls = ceil(self.config().num_hidden_layers / self.config().num_hidden_groups)
        super().test_context_simple(expected_number_of_adapter_calls=expected_number_of_adapter_calls)


@require_torch
@pytest.mark.embeddings
class Embeddings(
    AlbertAdapterTestBase,
    EmbeddingTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.class_conversion
class ClassConversion(
    ModelClassConversionTestMixin,
    AlbertAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.prefix_tuning
class PrefixTuning(
    AlbertAdapterTestBase,
    PrefixTuningTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.prompt_tuning
class PromptTuning(
    AlbertAdapterTestBase,
    PromptTuningTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.reft
class ReFT(
    AlbertAdapterTestBase,
    ReftTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.unipelt
class UniPELT(
    AlbertAdapterTestBase,
    UniPELTTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.compacter
class Compacter(
    AlbertAdapterTestBase,
    CompacterTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.bottleneck
class Bottleneck(
    AlbertAdapterTestBase,
    BottleneckAdapterTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.ia3
class IA3(
    AlbertAdapterTestBase,
    IA3TestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.lora
class LoRA(
    AlbertAdapterTestBase,
    LoRATestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.config_union
class ConfigUnion(
    AlbertAdapterTestBase,
    ConfigUnionAdapterTest,
    unittest.TestCase,
):
    pass
