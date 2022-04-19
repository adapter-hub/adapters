import torch

from transformers.adapters import PrefixTuningConfig
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

        input_data = self.get_input_samples((2, 128), config=model.config)

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
