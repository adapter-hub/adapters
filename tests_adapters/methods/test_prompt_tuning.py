import torch

from adapters import ADAPTER_MODEL_MAPPING, AutoAdapterModel, PrefixTuningConfig, PromptTuningConfig
from transformers.testing_utils import require_torch, torch_device

from .base import AdapterMethodBaseTestMixin


@require_torch
class PromptTuningTestMixin(AdapterMethodBaseTestMixin):
    def test_add_prompt_tuning(self):
        model = self.get_model()
        self.run_add_test(
            model, PromptTuningConfig(prompt_length=10), ["prompt_tunings.{name}."]
        )  # TODO: provide parameters in PromptTuningConfig(...) ?

    # TODO: add tests to add different configs (like initialization [random_uniform, from_array, ...] or prefix_prompt vs prefix_prompt_after_bos

    def test_average_prompt_tuning(self):
        model = self.get_model()
        self.run_average_test(model, PromptTuningConfig(prompt_length=10), ["prompt_tunings.{name}."])

    def test_delete_prompt_tuning(self):
        model = self.get_model()
        self.run_delete_test(model, PromptTuningConfig(prompt_length=10), ["prompt_tunings.{name}."])

    def test_get_prompt_tuning(self):
        model = self.get_model()
        self.run_get_test(
            model, PromptTuningConfig(prompt_length=10), 1
        )  # TODO: last number is number of layers. Is this really 1?

    def test_forward_prompt_tuning(self):
        model = self.get_model()
        self.run_forward_test(model, PromptTuningConfig(prompt_length=10))

    def test_load_prompt_tuning(self):
        self.run_load_test(PromptTuningConfig(prompt_length=10))

    def test_load_full_model_prefix_tuning(self):
        self.run_full_model_load_test(PromptTuningConfig(prompt_length=10))

    def test_train_prefix_tuning(self):
        self.run_train_test(PromptTuningConfig(prompt_length=10), ["prompt_tunings.{name}."])
