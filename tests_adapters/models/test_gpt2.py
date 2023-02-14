from tests.models.gpt2.test_modeling_gpt2 import *
from transformers import GPT2AdapterModel
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class GPT2AdapterModelTest(AdapterModelTesterMixin, GPT2ModelTest):
    all_model_classes = (GPT2AdapterModel,)
    fx_compatible = False
