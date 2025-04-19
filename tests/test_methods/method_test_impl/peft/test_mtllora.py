from adapters.configuration.adapter_config import MTLLoRAConfig
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch


@require_torch
class MTLLoRATestMixin(AdapterMethodBaseTestMixin):
    def test_add_mtllora(self):
        model = self.get_model()
        config = MTLLoRAConfig(n_up_projection=3)
        self.run_add_test(
            model,
            config,
            [
                "loras.{name}.",
            ],
        )

    def test_leave_out_mtllora(self):
        model = self.get_model()
        self.run_leave_out_test(model, MTLLoRAConfig(), self.leave_out_layers)

    def test_delete_mtllora(self):
        model = self.get_model()
        config = MTLLoRAConfig()
        self.run_delete_test(
            model,
            config,
            [
                "loras.{name}.",
            ],
        )

    def test_get_mtllora(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        self.run_get_test(
            model,
            MTLLoRAConfig(intermediate_lora=True, output_lora=True),
            n_layers * 3,
        )

    def test_forward_mtllora(self):
        model = self.get_model()
        for dtype in self.dtypes_to_test:
            for n_proj in [1, 3]:
                with self.subTest(
                    model_class=model.__class__.__name__,
                    dtype=dtype,
                    n_proj=n_proj,
                ):
                    self.run_forward_test(
                        model,
                        MTLLoRAConfig(
                            n_up_projection=n_proj,
                            init_weights="bert",  # avoid a zero tensor approx hidden state
                            intermediate_lora=True,
                            output_lora=True,
                        ),
                        dtype=dtype,
                    )

    def test_load_mtllora(self):
        self.run_load_test(MTLLoRAConfig())

    def test_load_full_model_mtllora(self):
        self.run_full_model_load_test(MTLLoRAConfig(init_weights="bert"))

    def test_train_mtllora(self):
        self.run_train_test(MTLLoRAConfig(init_weights="bert"), ["loras.{name}."])

    def test_mtllora_gradient_checkpointing_single_adapter(self):
        self.run_gradient_checkpointing_single_adapter_test(MTLLoRAConfig())
