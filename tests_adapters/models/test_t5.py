from tests.models.t5.test_modeling_t5 import *
from transformers import T5AdapterModel
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class T5AdapterModelTest(AdapterModelTesterMixin, T5ModelTest):
    all_model_classes = (T5AdapterModel,)
    fx_compatible = False
