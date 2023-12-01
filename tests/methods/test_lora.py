from adapters import LoRAConfig
from transformers.testing_utils import require_torch

from .base import AdapterMethodBaseTestMixin


@require_torch
class LoRATestMixin(AdapterMethodBaseTestMixin):
    def test_add_lora(self):
        model = self.get_model()
        self.run_add_test(model, LoRAConfig(), ["loras.{name}."])

    def test_leave_out_lora(self):
        model = self.get_model()
        self.run_leave_out_test(model, LoRAConfig(), self.leave_out_layers)

    def test_average_lora(self):
        model = self.get_model()
        self.run_average_test(model, LoRAConfig(), ["loras.{name}."])

    def test_delete_lora(self):
        model = self.get_model()
        self.run_delete_test(model, LoRAConfig(), ["loras.{name}."])

    def test_get_lora(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        self.run_get_test(model, LoRAConfig(intermediate_lora=True, output_lora=True), n_layers * 3)

    def test_forward_lora(self):
        model = self.get_model()
        self.run_forward_test(model, LoRAConfig(init_weights="bert", intermediate_lora=True, output_lora=True))

    def test_load_lora(self):
        self.run_load_test(LoRAConfig())

    def test_load_full_model_lora(self):
        self.run_full_model_load_test(LoRAConfig(init_weights="bert"))

    def test_train_lora(self):
        self.run_train_test(LoRAConfig(init_weights="bert"), ["loras.{name}."])

    def test_merge_lora(self):
        self.run_merge_test(LoRAConfig(init_weights="bert"))

    def test_reset_lora(self):
        self.run_reset_test(LoRAConfig(init_weights="bert"))
