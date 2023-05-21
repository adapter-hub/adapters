# flake8: noqa: F403,F405
from adapter_transformers import CLIPAdapterModel
from tests.models.clip.test_modeling_clip import *  # Imported to execute model tests
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class CLIPAdapterModelTest(AdapterModelTesterMixin, CLIPModelTest):
    all_model_classes = (CLIPAdapterModel,)
    fx_compatible = False
