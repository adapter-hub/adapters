import torch

from transformers.adapters import UniPELTConfig
from transformers.testing_utils import require_torch, torch_device

from .base import AdapterMethodBaseTestMixin


@require_torch
class UniPELTTestMixin(AdapterMethodBaseTestMixin):

    def test_add_unipelt(self):
        model = self.get_model()
        self.run_add_test(model, UniPELTConfig(), ["loras.{name}.", "adapters.{name}.", "prefix_tunings.{name}."])

    def test_delete_unipelt(self):
        model = self.get_model()
        self.run_delete_test(model, UniPELTConfig(), ["loras.{name}.", "adapters.{name}.", "prefix_tunings.{name}."])

    def test_get_unipelt(self):
        model = self.get_model()
        self.run_get_test(model, UniPELTConfig())

    def test_forward_unipelt(self):
        model = self.get_model()
        self.run_forward_test(model, UniPELTConfig())

    def test_load_unipelt(self):
        self.run_load_test(UniPELTConfig())

    def test_load_full_model_unipelt(self):
        self.run_full_model_load_test(UniPELTConfig())

    def test_train_unipelt(self):
        self.run_train_test(UniPELTConfig(), ["loras.{name}.", "adapters.{name}.", "prefix_tunings.{name}."])

    def test_merge_unipelt(self):
        self.run_merge_test(UniPELTConfig())

    def test_reset_unipelt(self):
        self.run_reset_test(UniPELTConfig())
