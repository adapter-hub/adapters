from adapters.configuration import (
    CompacterConfig,
    ConfigUnion,
    LoRAConfig,
    ParBnConfig,
    PrefixTuningConfig,
    SeqBnConfig,
)
from tests.methods.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch


@require_torch
class ConfigUnionAdapterTest(AdapterMethodBaseTestMixin):
    adapter_configs_to_test = [
        (
            ConfigUnion(
                PrefixTuningConfig(),
                ParBnConfig(),
            ),
            ["adapters.{name}.", "prefix_tunings.{name}."],
        ),
        (
            ConfigUnion(
                CompacterConfig(),
                LoRAConfig(),
            ),
            ["adapters.{name}.", "loras.{name}."],
        ),
        (
            ConfigUnion(
                SeqBnConfig(),
                LoRAConfig(),
            ),
            ["adapters.{name}.", "loras.{name}."],
        ),
    ]

    def test_add_union_adapter(self):
        model = self.get_model()
        model.eval()

        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_add_test(model, adapter_config, filter_keys)

    def test_union_adapter_forward(self):
        model = self.get_model()
        model.eval()

        for adapter_config, _ in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_forward_test(model, adapter_config)
