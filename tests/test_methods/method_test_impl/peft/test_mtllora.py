import random

import torch

from adapters import LoRAConfig
from adapters.configuration.adapter_config import MTLLoRAConfig
from adapters.methods.lora import LoRALayer
from transformers.testing_utils import require_torch

from .base import AdapterMethodBaseTestMixin


@require_torch
class MTLLoRATestMixin(AdapterMethodBaseTestMixin):
    def get_config(self, **kwargs):
        return MTLLoRAConfig(task_names=["t1", "t2"], **kwargs)

    def test_add_mtllora(self):
        model = self.get_model()
        self.run_add_test(
            model,
            self.get_config(),
            ["loras.{name}.", "loras.t1.", "loras.t2."],
        )

    def test_leave_out_mtllora(self):
        model = self.get_model()
        self.run_leave_out_test(model, self.get_config(), self.leave_out_layers)

    def test_delete_mtllora(self):
        model = self.get_model()
        self.run_delete_test(
            model,
            self.get_config(),
            ["loras.{name}.", "loras.t1.", "loras.t2."],
        )

    def test_get_mtllora(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        self.run_get_test(
            model,
            self.get_config(intermediate_lora=True, output_lora=True),
            n_layers * 3,
        )

    def test_forward_lora(self):
        model = self.get_model()
        for dtype in self.dtypes_to_test:
            with self.subTest(
                model_class=model.__class__.__name__, dtype=dtype
            ):
                self.run_forward_test(
                    model,
                    self.get_config(
                        init_weights="bert",
                        intermediate_lora=True,
                        output_lora=True,
                    ),
                    dtype=dtype,
                )

    def test_load_lora(self):
        self.run_load_test(self.get_config())

    def test_load_full_model_lora(self):
        self.run_full_model_load_test(self.get_config(init_weights="bert"))

    def test_train_lora(self):
        self.run_train_test(
            self.get_config(init_weights="bert"),
            ["loras.{name}.", "loras.t1.", "loras.t2."],
        )

    def test_merge_lora(self):
        self.run_merge_test(self.get_config(init_weights="bert"))

    def test_reset_lora(self):
        self.run_reset_test(self.get_config(init_weights="bert"))
