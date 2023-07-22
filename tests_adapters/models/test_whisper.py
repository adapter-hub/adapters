from tests.models.whisper.test_modeling_whisper import *
from transformers import WhisperAdapterModel
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class WhisperAdapterModelTest(AdapterModelTesterMixin, WhisperModelTest):
    all_model_classes = (WhisperAdapterModel,)
    fx_compatible = False
