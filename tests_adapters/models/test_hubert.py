# flake8: noqa: F403,F405
from adapters import HubertAdapterModel
from tests.models.test_wavlm import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class HubertAdapterModelTest(AdapterModelTesterMixin, HubertModelTest):
    all_model_classes = (HubertAdapterModel,)
    fx_compatible = False
