from tests.test_modeling_bert import *
from transformers import BertAdapterModel
from transformers.testing_utils import require_torch

from .test_common import AdapterModelTesterMixin


@require_torch
class BertAdapterModelTest(AdapterModelTesterMixin, BertModelTest):
    all_model_classes = (
        BertAdapterModel,
    )
