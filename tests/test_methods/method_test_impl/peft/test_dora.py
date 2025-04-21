from adapters import DoraConfig
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch


@require_torch
class DoraTestMixin(AdapterMethodBaseTestMixin):
    def test_add_Dora(self):
        model = self.get_model()
        self.run_add_test(model, DoraConfig(), ["loras.{name}."])

    def test_leave_out_Dora(self):
        model = self.get_model()
        self.run_leave_out_test(model, DoraConfig(), self.leave_out_layers)

    def test_linear_average_Dora(self):
        model = self.get_model()
        self.run_linear_average_test(model, DoraConfig(), ["loras.{name}."])

    def test_delete_Dora(self):
        model = self.get_model()
        self.run_delete_test(model, DoraConfig(), ["loras.{name}."])

    def test_get_Dora(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        self.run_get_test(model, DoraConfig(intermediate_lora=False, output_lora=False), n_layers)

    def test_forward_Dora(self):
        model = self.get_model()
        self.run_forward_test(model, DoraConfig(init_weights="lora", intermediate_lora=False, output_lora=False))

    def test_load_Dora(self):
        self.run_load_test(DoraConfig())

    def test_load_full_model_Dora(self):
        self.run_full_model_load_test(DoraConfig(init_weights="lora"))

    def test_train_Dora(self):
        self.run_train_test(DoraConfig(init_weights="lora"), ["loras.{name}."])

    def test_merge_Dora(self):
        self.run_merge_test(DoraConfig(init_weights="lora"))

    def test_reset_Dora(self):
        self.run_reset_test(DoraConfig(init_weights="lora"))

    def test_Dora_gradient_checkpointing_single_adapter(self):
        self.run_gradient_checkpointing_single_adapter_test(DoraConfig())

    def test_same_weights_after_adding_adapter(self):
        # setting init_weights_seed should leed to every adapter layer having the same weights after initialization
        self.run_same_weights_test(DoraConfig(init_weights_seed=42), ["loras.{name}."])
