import unittest
from dataclasses import FrozenInstanceError, dataclass

from transformers import ADAPTER_CONFIG_MAP, AdapterConfig, PfeifferConfig
from transformers.testing_utils import require_torch


@require_torch
class AdapterConfigTest(unittest.TestCase):

    config_names = ["pfeiffer", "houlsby"]

    def test_config_load(self):
        download_kwargs = {"force_download": True}
        for config_name in self.config_names:
            with self.subTest(config_name=config_name):
                config = AdapterConfig.load(config_name, download_kwargs=download_kwargs, non_linearity="leakyrelu")
                self.assertTrue(isinstance(config, AdapterConfig))
                self.assertEqual(config.non_linearity, "leakyrelu")

    def test_config_immutable(self):
        def set_attr(config: AdapterConfig):
            config.non_linearity = "dummy"

        for config in ADAPTER_CONFIG_MAP.values():
            with self.subTest(config=config.__class__.__name__):
                self.assertRaises(FrozenInstanceError, lambda: set_attr(config))

    def test_custom_attr(self):
        for config in ADAPTER_CONFIG_MAP.values():
            with self.subTest(config=config.__class__.__name__):
                # create a copy to leave original untouched
                config = config.replace()
                config.dummy_attr = "test_value"
                self.assertEqual(config.dummy_attr, "test_value")

    def test_custom_class(self):
        @dataclass
        class CustomAdapterConfig(PfeifferConfig):
            custom_attr: str = "test_value"

        config = CustomAdapterConfig()
        config_dict = config.to_dict()
        self.assertEqual(config_dict["custom_attr"], "test_value")
        # When calling load on an AdapterConfig instance, don't change the class of the config.
        config = AdapterConfig.load(config, custom_attr="test_value_2")
        self.assertTrue(isinstance(config, CustomAdapterConfig))
        self.assertEqual(config["custom_attr"], "test_value_2")
