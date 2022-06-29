import torch

from transformers.adapters import ADAPTER_MODEL_MAPPING, AutoAdapterModel, PrefixTuningConfig
from transformers.testing_utils import require_torch, torch_device

from .base import AdapterMethodBaseTestMixin


@require_torch
class PrefixTuningTestMixin(AdapterMethodBaseTestMixin):
    def test_add_prefix_tuning(self):
        model = self.get_model()
        self.run_add_test(model, PrefixTuningConfig(flat=True), ["prefix_tunings.{name}."])

    def test_delete_prefix_tuning(self):
        model = self.get_model()
        self.run_delete_test(model, PrefixTuningConfig(flat=True), ["prefix_tunings.{name}."])

    def test_get_prefix_tuning(self):
        model = self.get_model()
        self.run_get_test(model, PrefixTuningConfig(flat=True))

    def test_forward_prefix_tuning(self):
        model = self.get_model()
        self.run_forward_test(model, PrefixTuningConfig(flat=True))

    def test_load_prefix_tuning(self):
        self.run_load_test(PrefixTuningConfig())

    def test_train_prefix_tuning(self):
        self.run_train_test(PrefixTuningConfig(), ["prefix_tunings.{name}."])

    def test_eject_prefix(self):
        model = self.get_model()
        model.eval()
        model.add_adapter("test_prefix", config="prefix_tuning")
        model.to(torch_device)

        input_data = self.get_input_samples(config=model.config)

        # user reparamterized prefix
        model.set_active_adapters(["test_prefix"])
        output_1 = model(**input_data)

        # eject prefix
        model.eject_prefix_tuning("test_prefix")
        model.to(torch_device)
        model.eval()
        output_2 = model(**input_data)

        # check forward pass
        self.assertEqual(len(output_1), len(output_2))
        self.assertTrue(torch.allclose(output_1[0], output_2[0], atol=1e-4))

    def test_prefix_tuning_generate(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING or (
            not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_seq2seq_lm_head")
            and not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_causal_lm_head")
        ):
            self.skipTest("No seq2seq or causal language model head")

        model1 = AutoAdapterModel.from_config(self.config())
        model1.add_adapter("dummy", config="prefix_tuning")
        if hasattr(model1, "add_seq2seq_lm_head"):
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
