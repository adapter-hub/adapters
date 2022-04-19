from transformers.adapters import CompacterPlusPlusConfig
from transformers.testing_utils import require_torch

from .base import AdapterMethodBaseTestMixin


@require_torch
class CompacterTestMixin(AdapterMethodBaseTestMixin):
    def test_add_compacter(self):
        model = self.get_model()
        self.run_add_test(model, CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8), ["adapters.{name}."])

    def test_delete_compacter(self):
        model = self.get_model()
        self.run_delete_test(model, CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8), ["adapters.{name}."])

    def test_get_compacter(self):
        model = self.get_model()
        self.run_get_test(model, CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8))

    def test_forward_compacter(self):
        model = self.get_model()
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8)
        self.run_forward_test(model, adapter_config)

    def test_forward_shared_phm_compacter(self):
        model = self.get_model()
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, shared_W_phm=True, reduction_factor=8)
        self.run_forward_test(model, adapter_config)

    def test_load_compacter(self):
        self.run_load_test(CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8))

    def test_train_shared_w_compacter(self):
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, shared_W_phm=True, shared_phm_rule=False, reduction_factor=8)
        self.run_train_test(adapter_config, ["adapters.{name}."])

    def test_train_shared_phm_compacter(self):
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8)
        self.run_train_test(adapter_config, ["adapters.{name}."])
