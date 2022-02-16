from tests.test_modeling_mbart import *
from transformers import MBartAdapterModel
from transformers.testing_utils import require_torch

from .test_common import AdapterModelTesterMixin


@require_torch
class MBartAdapterModelTest(AdapterModelTesterMixin, MBartModelTest):
    all_model_classes = (
        MBartAdapterModel,
    )
