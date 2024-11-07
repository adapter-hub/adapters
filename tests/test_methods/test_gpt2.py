from transformers import GPT2Config

from .imports import *


class GPT2AdapterTestBase(TextAdapterTestBase):
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
@pytest.mark.core
class Core(
    GPT2AdapterTestBase,
    ModelClassConversionTestMixin,
    CompabilityTestMixin,
    AdapterFusionModelTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.composition
class Composition(
    GPT2AdapterTestBase,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    unittest.TestCase,
):
    def test_parallel_training_lora(self):
        self.skipTest("Not supported for GPT2")


@require_torch
@pytest.mark.heads
class Heads(
    GPT2AdapterTestBase,
    PredictionHeadModelTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.embeddings
class Embeddings(
    GPT2AdapterTestBase,
    EmbeddingTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.class_conversion
class ClassConversion(
    ModelClassConversionTestMixin,
    GPT2AdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.prefix_tuning
class PrefixTuning(
    GPT2AdapterTestBase,
    PrefixTuningTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.prompt_tuning
class PromptTuning(
    GPT2AdapterTestBase,
    PromptTuningTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.reft
class ReFT(
    GPT2AdapterTestBase,
    ReftTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.unipelt
class UniPELT(
    GPT2AdapterTestBase,
    UniPELTTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.compacter
class Compacter(
    GPT2AdapterTestBase,
    CompacterTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.bottleneck
class Bottleneck(
    GPT2AdapterTestBase,
    BottleneckAdapterTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.ia3
class IA3(
    GPT2AdapterTestBase,
    IA3TestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.lora
class LoRA(
    GPT2AdapterTestBase,
    LoRATestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.config_union
class ConfigUnion(
    GPT2AdapterTestBase,
    ConfigUnionAdapterTest,
    unittest.TestCase,
):
    pass
