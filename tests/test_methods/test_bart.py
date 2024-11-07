from transformers import BartConfig

from .imports import *


class BartAdapterTestBase(TextAdapterTestBase):
    config_class = BartConfig
    config = make_config(
        BartConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
    )
    tokenizer_name = "facebook/bart-base"


@require_torch
@pytest.mark.core
class Core(
    BartAdapterTestBase,
    CompabilityTestMixin,
    AdapterFusionModelTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.composition
class Composition(
    BartAdapterTestBase,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.heads
class Heads(
    BartAdapterTestBase,
    PredictionHeadModelTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.embeddings
class Embeddings(
    BartAdapterTestBase,
    EmbeddingTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
class BartClassConversionTest(
    ModelClassConversionTestMixin,
    BartAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.prefix_tuning
class PrefixTuning(
    BartAdapterTestBase,
    PrefixTuningTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.reft
class ReFT(
    BartAdapterTestBase,
    ReftTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.unipelt
class UniPELT(
    BartAdapterTestBase,
    UniPELTTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.compacter
class Compacter(
    BartAdapterTestBase,
    CompacterTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.bottleneck
class Bottleneck(
    BartAdapterTestBase,
    BottleneckAdapterTestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.ia3
class IA3(
    BartAdapterTestBase,
    IA3TestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.lora
class LoRA(
    BartAdapterTestBase,
    LoRATestMixin,
    unittest.TestCase,
):
    pass


@require_torch
@pytest.mark.config_union
class ConfigUnion(
    BartAdapterTestBase,
    ConfigUnionAdapterTest,
    unittest.TestCase,
):
    pass
