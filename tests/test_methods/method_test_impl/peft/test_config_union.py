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
                ParBnConfig(),
            ),
            ["adapters.{name}.", "prefix_tunings.{name}."],
        ),
        (
            ConfigUnion(
                CompacterConfig(
                    reduction_factor=8
                ),  # set to smaller value than default due to smaller hidden size of test models
                LoRAConfig(init_weights="bert"),  # set to bert to avoid zero initialization
            ),
            ["adapters.{name}.", "loras.{name}."],
        ),
        (
            ConfigUnion(
                SeqBnConfig(phm_dim=1),
                LoRAConfig(init_weights="bert"),  # set to bert to avoid zero initialization
            ),
            ["adapters.{name}.", "loras.{name}."],
        ),
    ]

    def test_add_union_adapter(self):
        model = self.get_model()
        model.eval()
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            config = (
                "ConfigUnion: "
                + adapter_config.configs[0].__class__.__name__
                + adapter_config.configs[1].__class__.__name__
            )
            with self.subTest(model_class=model.__class__.__name__, config=config):
                self.run_add_test(model, adapter_config, filter_keys)

    def test_union_adapter_forward(self):
        model = self.get_model()
        model.eval()
        for adapter_config, _ in self.adapter_configs_to_test:
            config = (
                "ConfigUnion: "
                + adapter_config.configs[0].__class__.__name__
                + adapter_config.configs[1].__class__.__name__
            )
            with self.subTest(model_class=model.__class__.__name__, config=config):
                self.run_forward_test(model, adapter_config)
