from tests.test_modeling_distilbert import *
from transformers import DistilBertAdapterModel
from transformers.testing_utils import require_torch

from .test_common import AdapterModelTesterMixin


@require_torch
class DistilBertAdapterModelTest(AdapterModelTesterMixin, DistilBertModelTest):
    all_model_classes = (
        DistilBertAdapterModel,
    )
