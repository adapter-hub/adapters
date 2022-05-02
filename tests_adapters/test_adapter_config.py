import json
import unittest
from dataclasses import FrozenInstanceError, dataclass

from transformers import (
    ADAPTER_CONFIG_MAP,
    AdapterConfig,
    AdapterConfigBase,
    ConfigUnion,
    HoulsbyConfig,
    MAMConfig,
    ParallelConfig,
    PfeifferConfig,
    PrefixTuningConfig,
)
from transformers.testing_utils import require_torch


@require_torch
class AdapterConfigTest(unittest.TestCase):
    def test_config_load(self):
        download_kwargs = {"force_download": True}
        for config_name in ["pfeiffer", "houlsby"]:
            with self.subTest(config_name=config_name):
                config = AdapterConfig.load(config_name, download_kwargs=download_kwargs, non_linearity="leakyrelu")
                self.assertTrue(isinstance(config, AdapterConfig))
                self.assertEqual(config.non_linearity, "leakyrelu")

    def test_config_immutable(self):
        def set_attr(config: AdapterConfig):
            config.non_linearity = "dummy"
            config.r = -1  # for LoRA

        for config in ADAPTER_CONFIG_MAP.values():
            if isinstance(config, ConfigUnion):
                continue
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

    def test_config_union_valid(self):
        unions = [
            [PrefixTuningConfig(), ParallelConfig()],
            [PrefixTuningConfig(), PfeifferConfig()],
            [HoulsbyConfig(mh_adapter=False), HoulsbyConfig(output_adapter=False, reduction_factor=2)],
            [PfeifferConfig(leave_out=[9, 10, 11], reduction_factor=2), PfeifferConfig(leave_out=list(range(9)))],
        ]
        for union in unions:
            with self.subTest(union=union):
                config = ConfigUnion(*union)
                self.assertEqual(config.architecture, "union")

                # make sure serialization/ deserialization works
                config_dict = config.to_dict()
                config_dict = json.loads(json.dumps(config_dict))
                config_new = ConfigUnion.from_dict(config_dict)
                self.assertEqual(config, config_new)

                self.assertIsInstance(config_new[0], AdapterConfigBase)
                self.assertIsInstance(config_new[1], AdapterConfigBase)

    def test_config_union_invalid(self):
        unions = [
            ([MAMConfig(), PfeifferConfig()], TypeError),
            ([PfeifferConfig(), PfeifferConfig()], ValueError),
            ([PfeifferConfig(), HoulsbyConfig()], ValueError),
        ]
        for union, error_type in unions:
            with self.subTest(union=union):
                self.assertRaises(error_type, ConfigUnion, *union)
