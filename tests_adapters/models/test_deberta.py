from tests.models.deberta.test_modeling_deberta import *
from transformers import DebertaAdapterModel
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class DebertaAdapterModelTest(AdapterModelTesterMixin, DebertaModelTest):
    all_model_classes = (DebertaAdapterModel,)
    fx_compatible = False
