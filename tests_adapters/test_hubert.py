import unittest

from tests_adapters.methods.test_config_union import ConfigUnionAdapterTest
from transformers import HubertConfig
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


class HubertAdapterTestBase(AdapterTestBase):
    config_class = HubertConfig
    config = make_config(
        HubertConfig,
        # hidden_size=32,
        # num_hidden_layers=5,
        # num_attention_heads=4,
        # intermediate_size=37,
        # hidden_act="gelu",
        # relative_attention=True,
        # pos_att_type="p2c|c2p",
    )
    feature_extractor_name = "facebook/hubert-base-ls960"


@require_torch
class HubertAdapterTest(
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    EmbeddingTestMixin,
    ParallelTrainingMixin,
    ConfigUnionAdapterTest,
    HubertAdapterTestBase,
    unittest.TestCase,
):
    pass

@require_torch
class DebertaClassConversionTest(
    ModelClassConversionTestMixin,
    HubertAdapterTestBase,
    unittest.TestCase,
):
    pass
