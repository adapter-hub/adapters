from tests.test_modeling_roberta import *
from transformers import RobertaAdapterModel
from transformers.testing_utils import require_torch

from .test_common import AdapterModelTesterMixin


@require_torch
class RobertaAdapterModelTest(AdapterModelTesterMixin, RobertaModelTest):
    all_model_classes = (
        RobertaAdapterModel,
    )
