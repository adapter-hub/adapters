from adapters import ADAPTER_MODEL_MAPPING, AutoAdapterModel, CompacterPlusPlusConfig
from transformers.testing_utils import require_torch, torch_device

from .base import AdapterMethodBaseTestMixin


@require_torch
class CompacterTestMixin(AdapterMethodBaseTestMixin):
    def test_add_compacter(self):
        model = self.get_model()
        self.run_add_test(model, CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8), ["adapters.{name}."])

    def test_leave_out_compacter(self):
        model = self.get_model()
        self.run_leave_out_test(model, CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8), self.leave_out_layers)

    def test_average_compacter(self):
        model = self.get_model()
        self.run_average_test(model, CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8), ["adapters.{name}."])

    def test_delete_compacter(self):
        model = self.get_model()
        self.run_delete_test(model, CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8), ["adapters.{name}."])

    def test_get_compacter(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        self.run_get_test(model, CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8), n_layers + 1)

    def test_forward_compacter(self):
        model = self.get_model()
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8)
        self.run_forward_test(model, adapter_config)

    def test_forward_shared_phm_compacter(self):
        model = self.get_model()
        adapter_config = CompacterPlusPlusConfig(phm_dim=4, shared_W_phm=True, reduction_factor=4)
        self.run_forward_test(model, adapter_config)

    def test_load_compacter(self):
        self.run_load_test(CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8))

    def test_train_shared_w_compacter(self):
        adapter_config = CompacterPlusPlusConfig(
            phm_dim=2, shared_W_phm=True, shared_phm_rule=False, reduction_factor=8
        )
        self.run_train_test(adapter_config, ["adapters.{name}."])

    def test_train_shared_phm_compacter(self):
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8)
        self.run_train_test(adapter_config, ["adapters.{name}."])

    def test_compacter_generate(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING or (
            "seq2seq_lm" not in ADAPTER_MODEL_MAPPING[self.config_class].head_types
            and "causal_lm" not in ADAPTER_MODEL_MAPPING[self.config_class].head_types
        ):
            self.skipTest("No seq2seq or causal language model head")

        model1 = AutoAdapterModel.from_config(self.config())
        model1.add_adapter("dummy", config=CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8))
        if "seq2seq_lm" in ADAPTER_MODEL_MAPPING[self.config_class].head_types:
            model1.add_seq2seq_lm_head("dummy")
        else:
            model1.add_causal_lm_head("dummy")
        model1.set_active_adapters("dummy")
        model1.to(torch_device)

        seq_output_length = 32

        # Finally, also check if generation works properly
        input_ids = self.get_input_samples((1, 4), config=model1.config)["input_ids"]
        input_ids = input_ids.to(torch_device)
        generated = model1.generate(input_ids, max_length=seq_output_length)
        self.assertLessEqual(generated.shape, (1, seq_output_length))
