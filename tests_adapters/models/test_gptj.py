from tests.models.gptj.test_modeling_gptj import *
from transformers import GPTJAdapterModel
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class GPTJAdapterModelTest(AdapterModelTesterMixin, GPTJModelTest):
    all_model_classes = (GPTJAdapterModel,)
    fx_compatible = False
