import unittest

from tests.bert.test_modeling_bert import *
from transformers import BertAdapterModel
from transformers.testing_utils import require_torch

from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_common import AdapterModelTestMixin
from .test_adapter_compacter import CompacterTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_adapter_training import AdapterTrainingTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class BertAdapterModelTest(AdapterModelTesterMixin, BertModelTest):
    all_model_classes = (
        BertAdapterModel,
    )


class BertAdapterTestBase(AdapterTestBase):
    config_class = BertConfig
    config = make_config(
        BertConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    tokenizer_name = "bert-base-uncased"


@require_torch
class BertAdapterTest(
    EmbeddingTestMixin,
    CompacterTestMixin,
    AdapterModelTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    AdapterTrainingTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    BertAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class BertClassConversionTest(
    ModelClassConversionTestMixin,
    BertAdapterTestBase,
    unittest.TestCase,
):
    pass
