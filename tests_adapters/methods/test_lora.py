import torch

from transformers.adapters import LoRAConfig
from transformers.testing_utils import require_torch, torch_device

from .base import AdapterMethodBaseTestMixin


@require_torch
class LoRATestMixin(AdapterMethodBaseTestMixin):

    def test_add_lora(self):
        model = self.get_model()
        self.run_add_test(model, LoRAConfig(), ["loras.{name}."])

    def test_delete_lora(self):
        model = self.get_model()
        self.run_delete_test(model, LoRAConfig(), ["loras.{name}."])

    def test_get_lora(self):
        model = self.get_model()
        self.run_get_test(model, LoRAConfig())

    def test_forward_lora(self):
        model = self.get_model()
        self.run_forward_test(model, LoRAConfig(init_weights="bert", intermediate_lora=True, output_lora=True))

    def test_load_lora(self):
        self.run_load_test(LoRAConfig())

    def test_train_lora(self):
        self.run_train_test(LoRAConfig(init_weights="bert"), ["loras.{name}."])

    def test_merge_lora(self):
        model = self.get_model()
        model.eval()
        model.add_adapter("test_lora", config=LoRAConfig(init_weights="bert"))
        model.to(torch_device)

        input_data = self.get_input_samples(config=model.config)

        # forward in training mode
        model.set_active_adapters(["test_lora"])
        output_1 = model(**input_data)

        # forward in merged mode
        model.set_active_adapters(None)
        model.merge_lora("test_lora")
        model.to(torch_device)
        model.eval()
        output_2 = model(**input_data)

        # check forward pass
        self.assertEqual(len(output_1), len(output_2))
        self.assertTrue(torch.allclose(output_1[0], output_2[0], atol=1e-3))

    def test_reset_lora(self):
        model = self.get_model()
        model.eval()
        model.add_adapter("test_lora", config=LoRAConfig(init_weights="bert"))
        model.to(torch_device)

        input_data = self.get_input_samples(config=model.config)

        # before merging
        output_1 = model(**input_data)

        # merge & reset
        model.merge_lora("test_lora")
        model.reset_lora()

        # after merging
        output_2 = model(**input_data)

        # check forward pass
        self.assertEqual(len(output_1), len(output_2))
        self.assertTrue(torch.allclose(output_1[0], output_2[0], atol=1e-3))
