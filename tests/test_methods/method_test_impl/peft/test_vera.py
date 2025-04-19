from adapters import VeraConfig
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch


@require_torch
class VeraTestMixin(AdapterMethodBaseTestMixin):
    def test_add_Vera(self):
        model = self.get_model()
        # don't include "shared_parameters.{name}." here since they are frozen and this test checks if the adapter weights are active.
        self.run_add_test(model, VeraConfig(), ["loras.{name}."])

    def test_leave_out_Vera(self):
        model = self.get_model()
        self.run_leave_out_test(model, VeraConfig(), self.leave_out_layers)

    def test_linear_average_Vera(self):
        model = self.get_model()
        self.run_linear_average_test(model, VeraConfig(), ["loras.{name}.", "shared_parameters.{name}."])

    def test_delete_Vera(self):
        model = self.get_model()
        self.run_delete_test(model, VeraConfig(), ["loras.{name}.", "shared_parameters.{name}."])

    def test_get_Vera(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        self.run_get_test(model, VeraConfig(intermediate_lora=False, output_lora=False), n_layers + 1)

    def test_forward_Vera(self):
        model = self.get_model()
        self.run_forward_test(
            model, VeraConfig(init_weights="vera", intermediate_lora=False, output_lora=False, vera_b=0.5, vera_d=0.5)
        )

    def test_load_Vera(self):
        self.run_load_test(VeraConfig())

    def test_load_full_model_Vera(self):
        self.run_full_model_load_test(VeraConfig(init_weights="vera"))

    def test_train_Vera(self):
        # don't include "shared_parameters.{name}." here since they are frozen and this test checks if the adapter weights are active.
        self.run_train_test(VeraConfig(init_weights="vera"), ["loras.{name}."])

    def test_merge_Vera(self):
        self.run_merge_test(VeraConfig(init_weights="vera"))

    def test_reset_Vera(self):
        self.run_reset_test(VeraConfig(init_weights="vera"))

    def test_Vera_gradient_checkpointing_single_adapter(self):
        self.run_gradient_checkpointing_single_adapter_test(VeraConfig())

    def test_same_weights_after_adding_adapter(self):
        # setting init_weights_seed should leed to every adapter layer having the same weights after initialization
        self.run_same_weights_test(VeraConfig(init_weights_seed=42), ["loras.{name}.", "shared_parameters.{name}."])

    def test_vera_unsupported_combine_strategies(self):
        # VeRA only supports linear averaging. (see https://docs.adapterhub.ml/merging_adapters.html)

        model = self.get_model()
        model.eval()

        model.add_adapter("test_adapter_1", config=VeraConfig(init_weights="vera"))
        model.add_adapter("test_adapter_2", config=VeraConfig(init_weights="vera"))

        # First, check "lora_linear_only_negate_b"
        with self.assertRaisesRegex(ValueError, "VeRA only supports linear averaging"):
            model.average_adapter(
                "merged_adapter_name",
                ["test_adapter_1", "test_adapter_2"],
                weights=[0.1, 0.9],
                combine_strategy="lora_linear_only_negate_b",
            )

        # Next, check "lora_delta_w_svd"
        expected_error = (
            "This model specifically does not support 'lora_delta_w_svd' as a merging method. Please use a different combine_strategy or a different model."
            if model.config.model_type == "deberta-v2"
            else "VeRA only supports linear averaging"
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            model.average_adapter(
                "merged_adapter_name",
                ["test_adapter_1", "test_adapter_2"],
                weights=[0.1, 0.9],
                combine_strategy="lora_delta_w_svd",
                svd_rank=1,
            )
