from tests.test_modeling_bart import *
from transformers import BartAdapterModel
from transformers.testing_utils import require_torch

from .test_common import AdapterModelTesterMixin


@require_torch
class BartAdapterModelTest(AdapterModelTesterMixin, BartModelTest):
    all_model_classes = (
        BartAdapterModel,
    )
