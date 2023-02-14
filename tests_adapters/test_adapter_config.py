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
    LoRAConfig,
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

    def test_config_string_valid(self):
        to_test = [
            ("houlsby", HoulsbyConfig()),
            ("pfeiffer[reduction_factor=2, leave_out=[11]]", PfeifferConfig(reduction_factor=2, leave_out=[11])),
            ("parallel[reduction_factor={'0': 8, '1': 8, 'default': 16}]", ParallelConfig(reduction_factor={"0": 8, "1": 8, "default": 16})),
            ("prefix_tuning[prefix_length=30, flat=True]", PrefixTuningConfig(prefix_length=30, flat=True)),
            ("lora[r=200,alpha=8]", LoRAConfig(r=200, alpha=8)),
            ("prefix_tuning|parallel", ConfigUnion(PrefixTuningConfig(), ParallelConfig())),
            ("lora[attn_matrices=['k', 'v']]", LoRAConfig(attn_matrices=["k", "v"])),
            ("lora[use_gating=True]|prefix_tuning[use_gating=True]|pfeiffer[use_gating=True]", ConfigUnion(LoRAConfig(use_gating=True), PrefixTuningConfig(use_gating=True), PfeifferConfig(use_gating=True))),
        ]
        for config_str, config in to_test:
            with self.subTest(config_str=config_str):
                config_new = AdapterConfig.load(config_str)
                self.assertEqual(config, config_new)

    def test_config_string_invalid(self):
        to_test = [
            ("pfeiffer[invalid_key=2]", TypeError),
            ("lora[r=8]|invalid_name", ValueError),
            ("prefix_tuning[flat=True", ValueError),
            ("houlsby[reduction_factor=dict(default=1)]", ValueError),
        ]
        for config_str, error_type in to_test:
            with self.subTest(config_str=config_str):
                self.assertRaises(error_type, AdapterConfig.load, config_str)
