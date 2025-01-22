from adapters.configuration import (
    CompacterConfig,
    ConfigUnion,
    LoRAConfig,
    ParBnConfig,
    PrefixTuningConfig,
    SeqBnConfig,
)
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch


@require_torch
class ConfigUnionAdapterTest(AdapterMethodBaseTestMixin):
    adapter_configs_to_test = [
        (
            ConfigUnion(
                PrefixTuningConfig(),
                ParBnConfig(phm_dim=1),
            ),
            ["adapters.{name}.", "prefix_tunings.{name}."],
        ),
    ]
    adapter_configs_to_debug = [
        (
            ConfigUnion(
                SeqBnConfig(phm_dim=1),
                LoRAConfig(),
            ),
            ["adapters.{name}.", "loras.{name}."],
        ),
        (
            ConfigUnion(
                LoRAConfig(),
                CompacterConfig(phm_dim=1),
            ),
            ["adapters.{name}.", "loras.{name}."],
        ),
    ]

    def test_add_union_adapter(self):
        for adapter_config, filter_keys in self.adapter_configs_to_debug:
            model = self.get_model()
            model.eval()
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_add_test(model, adapter_config, filter_keys)

    def test_union_adapter_forward(self):
        for adapter_config, _ in self.adapter_configs_to_debug:
            model = self.get_model()
            model.eval()
            adapter_name = adapter_config.configs[0].__class__.__name__ + adapter_config.configs[1].__class__.__name__
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                print(f"Testing config: {adapter_name}")
                self.run_forward_test(model, adapter_config)
