import unittest

from tests.models.albert.test_modeling_albert import *
from transformers import AlbertAdapterModel
from transformers.testing_utils import require_torch

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
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin
from math import ceil


@require_torch
class AlbertAdapterModelTest(AdapterModelTesterMixin, AlbertModelTest):
    all_model_classes = (AlbertAdapterModel,)
    fx_compatible = False


class AlbertAdapterTestBase(AdapterTestBase):
    config_class = AlbertConfig
    config = make_config(
        AlbertConfig,
        embedding_size=16,
        hidden_size=256,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        num_hidden_groups=2,
    )
    tokenizer_name = "albert-base-v2"


@require_torch
class AlbertAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    EmbeddingTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    AlbertAdapterTestBase,
    unittest.TestCase,
):
    def test_context_simple(self):
        expected_number_of_adapter_calls = ceil(self.config().num_hidden_layers / self.config().num_hidden_groups)
        super().test_context_simple(expected_number_of_adapter_calls=expected_number_of_adapter_calls)

    def test_add_embeddings(self):
        super().test_add_embeddings(embedding_dim=self.config().embedding_size)

    def test_add_embedding_tokens(self):
        super().test_add_embedding_tokens(embedding_dim=self.config().embedding_size)

    def test_delete_embeddings(self):
        super().test_delete_embeddings(embedding_dim=self.config().embedding_size)

    def test_save_load_embedding(self):
        super().test_save_load_embedding(embedding_dim=self.config().embedding_size)

    def test_training_embedding(self):
        super().test_training_embedding(embedding_dim=self.config().embedding_size)

    def test_reference_embedding(self):
        super().test_reference_embedding(embedding_dim=self.config().embedding_size)


@require_torch
class AlbertClassConversionTest(
    ModelClassConversionTestMixin,
    AlbertAdapterTestBase,
    unittest.TestCase,
):
    pass
