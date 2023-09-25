# flake8: noqa: F403,F405
from adapters import WavLMAdapterModel
from hf_transformers.tests.models.wavlm.test_modeling_wavlm  import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class WavLMAdapterModelTest(AdapterModelTesterMixin, WavLMModelTest):
    all_model_classes = (WavLMAdapterModel,)
    fx_compatible = False
