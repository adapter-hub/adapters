from tests.models.mbart.test_modeling_mbart import *
from transformers import MBartAdapterModel
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class MBartAdapterModelTest(AdapterModelTesterMixin, MBartModelTest):
    all_model_classes = (MBartAdapterModel,)
    fx_compatible = False
