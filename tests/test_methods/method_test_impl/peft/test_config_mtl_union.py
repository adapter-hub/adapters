from adapters.composition import MultiTask
from adapters.configuration.adapter_config import ConfigMultiTaskUnion, MTLLoRAConfig
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch


@require_torch
class ConfigMultiTaskUnionAdapterTest(AdapterMethodBaseTestMixin):

    adapter_configs_to_test = [
        (
            ConfigMultiTaskUnion(base_config=MTLLoRAConfig(), task_names=["a", "b", "c"]),
            [
                "loras.shared_parameters.{name}.",
                "loras.a.",
                "loras.b.",
                "loras.c.",
            ],
        ),
    ]

    def test_add_mtl_union_adapter(self):
        model = self.get_model()
        model.eval()
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            config = "MTLConfigUnion: " + adapter_config.base_config.__class__.__name__
            with self.subTest(
                model_class=model.__class__.__name__,
                config=config,
                task_names=adapter_config.task_names,
            ):
                self.run_add_test(model, adapter_config, filter_keys)

    def test_mtl_union_adapter_forward(self):
        model = self.get_model()
        model.eval()
        for adapter_config, _ in self.adapter_configs_to_test:
            config = "MTLConfigUnion: " + adapter_config.base_config.__class__.__name__
            with self.subTest(
                model_class=model.__class__.__name__,
                config=config,
                task_names=adapter_config.task_names,
            ):
                self.run_forward_test(
                    model,
                    adapter_config,
                    adapter_setup=MultiTask(*adapter_config.task_names),
                )
