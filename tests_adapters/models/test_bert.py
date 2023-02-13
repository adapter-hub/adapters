from tests.models.bert.test_modeling_bert import *
from transformers import BertAdapterModel
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class BertAdapterModelTest(AdapterModelTesterMixin, BertModelTest):
    all_model_classes = (BertAdapterModel,)
    fx_compatible = False
