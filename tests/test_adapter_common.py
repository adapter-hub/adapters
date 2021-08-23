import copy
import tempfile

import torch

from transformers import (
    ADAPTER_CONFIG_MAP,
    MODEL_WITH_HEADS_MAPPING,
    AutoModelWithHeads,
    HoulsbyConfig,
    HoulsbyInvConfig,
    PfeifferConfig,
    PfeifferInvConfig,
)
from transformers.testing_utils import require_torch


def create_twin_models(model_class, config_creator=None):
    if config_creator and model_class.__name__.startswith("Auto"):
        model_config = config_creator()
        model1 = model_class.from_config(model_config)
    elif config_creator:
        model_config = config_creator()
        model1 = model_class(model_config)
    else:
        model_config = model_class.config_class()
        model1 = model_class(model_config)
    model1.eval()
    # create a twin initialized with the same random weights
    model2 = copy.deepcopy(model1)
    model2.eval()
    return model1, model2


@require_torch
class AdapterModelTestMixin:
    def test_add_adapter(self):
        model = self.get_model()
        model.eval()

        for adapter_config in [PfeifferConfig(), HoulsbyConfig()]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config)
                model.set_active_adapters([name])

                # adapter is correctly added to config
                self.assertTrue(name in model.config.adapters)
                self.assertEqual(adapter_config, model.config.adapters.get(name))

                # check forward pass
                input_data = self.get_input_samples((1, 128), config=model.config)
                adapter_output = model(**input_data)
                model.set_active_adapters(None)
                base_output = model(**input_data)
                self.assertEqual(len(adapter_output), len(base_output))
                self.assertFalse(torch.equal(adapter_output[0], base_output[0]))

    def test_delete_adapter(self):
        model = self.get_model()
        model.eval()

        name = "test_adapter"
        model.add_adapter(name, config="houlsby")
        model.set_active_adapters([name])

        # adapter is correctly added to config
        self.assertTrue(name in model.config.adapters)
        self.assertGreater(len(model.get_adapter(name)), 0)

        # remove the adapter again
        model.delete_adapter(name)
        self.assertFalse(name in model.config.adapters)
        self.assertEqual(len(model.get_adapter(name)), 0)

    def test_add_adapter_with_invertible(self):
        model = self.get_model()
        model.eval()

        for adapter_config in [PfeifferInvConfig(), HoulsbyInvConfig()]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config)
                model.set_active_adapters([name])

                # adapter is correctly added to config
                self.assertTrue(name in model.config.adapters)
                self.assertEqual(adapter_config, model.config.adapters.get(name))

                # invertible adapter is correctly added and returned
                self.assertTrue(name in model.invertible_adapters)
                self.assertEqual(model.invertible_adapters[name], model.get_invertible_adapter())

                # all invertible adapter weights should be activated for training
                for param in model.invertible_adapters[name].parameters():
                    self.assertTrue(param.requires_grad)

                # check forward pass
                input_data = self.get_input_samples((1, 128), config=model.config)
                adapter_output = model(**input_data)
                # make sure the output is different without invertible adapter
                del model.invertible_adapters[name]
                adapter_output_no_inv = model(**input_data)
                self.assertEqual(len(adapter_output), len(adapter_output_no_inv))
                self.assertFalse(torch.equal(adapter_output[0], adapter_output_no_inv[0]))

    def test_get_adapter(self):
        model = self.get_model()
        model.eval()

        adapter_config = HoulsbyConfig()
        model.add_adapter("first", config=adapter_config)
        model.add_adapter("second", config=adapter_config)
        model.set_active_adapters(["first"])

        # adapter is correctly added to config
        name = "first"
        self.assertTrue(name in model.config.adapters)
        self.assertEqual(adapter_config, model.config.adapters.get(name))

        first_adapter = model.get_adapter("first")
        second_adapter = model.get_adapter("second")

        self.assertNotEqual(len(first_adapter), 0)
        self.assertEqual(len(first_adapter), len(second_adapter))
        self.assertNotEqual(first_adapter, second_adapter)

    def test_add_adapter_multiple_reduction_factors(self):
        model = self.get_model()
        model.eval()
        reduction_factor = {"1": 1, "default": 2}
        for adapter_config in [
            PfeifferConfig(reduction_factor=reduction_factor),
            HoulsbyConfig(reduction_factor=reduction_factor),
        ]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config)
                model.set_active_adapters([name])

                # adapter is correctly added to config
                self.assertTrue(name in model.config.adapters)
                self.assertEqual(adapter_config, model.config.adapters.get(name))

                adapter = model.get_adapter(name)

                self.assertEqual(
                    adapter[0]["output"].adapter_down[0].in_features
                    / adapter[0]["output"].adapter_down[0].out_features,
                    reduction_factor["default"],
                )
                self.assertEqual(
                    adapter[1]["output"].adapter_down[0].in_features
                    / adapter[1]["output"].adapter_down[0].out_features,
                    reduction_factor["1"],
                )

    def test_reduction_factor_no_default(self):
        model = self.get_model()
        model.eval()
        reduction_factor = {"2": 8, "4": 32}
        for adapter_config in [
            PfeifferConfig(reduction_factor=reduction_factor),
            HoulsbyConfig(reduction_factor=reduction_factor),
        ]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                with self.assertRaises(KeyError):
                    model.add_adapter(name, config=adapter_config)

    def test_adapter_forward(self):
        model = self.get_model()
        model.eval()

        for adapter_config in [PfeifferConfig(), HoulsbyConfig()]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config)

                input_data = self.get_input_samples((1, 128), config=model.config)

                # set via property
                model.set_active_adapters([name])
                output_1 = model(**input_data)

                # unset and make sure it's unset
                model.set_active_adapters(None)
                self.assertEqual(None, model.active_adapters)

                # check forward pass
                output_2 = model(**input_data, adapter_names=[name])
                self.assertEqual(len(output_1), len(output_2))
                self.assertTrue(torch.equal(output_1[0], output_2[0]))

    def test_load_adapter(self):
        model1, model2 = create_twin_models(self.model_class, self.config)

        name = "dummy"
        model1.add_adapter(name)
        model1.set_active_adapters([name])
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter(temp_dir, name)

            # also tests that set_active works
            model2.load_adapter(temp_dir, set_active=True)

        # check if adapter was correctly loaded
        self.assertTrue(name in model2.config.adapters)

        # check equal output
        input_data = self.get_input_samples((1, 128), config=model1.config)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_load_full_model(self):
        model1 = self.get_model()
        model1.eval()

        name = "dummy"
        model1.add_adapter(name)
        model1.set_active_adapters([name])
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_pretrained(temp_dir)

            model2 = self.model_class.from_pretrained(temp_dir)
            model2.set_active_adapters([name])

        # check if adapter was correctly loaded
        self.assertTrue(name in model2.config.adapters)

        # check equal output
        input_data = self.get_input_samples((1, 128), config=model1.config)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_model_config_serialization(self):
        """PretrainedConfigurations should not raise an Exception when serializing the config dict

        See, e.g., PretrainedConfig.to_json_string()
        """
        for k, v in ADAPTER_CONFIG_MAP.items():
            model = self.get_model()
            model.add_adapter("test", config=v)
            # should not raise an exception
            model.config.to_json_string()

    def test_loading_adapter_weights_with_prefix(self):
        if self.config_class not in MODEL_WITH_HEADS_MAPPING:
            self.skipTest("Does not support flex heads.")

        model_base, model_with_head_base = create_twin_models(self.model_class, self.config)

        model_with_head = AutoModelWithHeads.from_config(model_with_head_base.config)
        setattr(model_with_head, model_with_head.base_model_prefix, model_with_head_base)

        model_with_head.add_adapter("dummy")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_with_head.save_adapter(temp_dir, "dummy")

            loading_info = {}
            model_base.load_adapter(temp_dir, loading_info=loading_info)

        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # check equal output
        input_data = self.get_input_samples((1, 128), config=model_with_head.config)
        output1 = model_with_head(**input_data)
        output2 = model_base(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_loading_adapter_weights_without_prefix(self):
        if self.config_class not in MODEL_WITH_HEADS_MAPPING:
            self.skipTest("Does not support flex heads.")

        model_base, model_with_head_base = create_twin_models(self.model_class, self.config)

        model_with_head = AutoModelWithHeads.from_config(model_with_head_base.config)
        setattr(model_with_head, model_with_head.base_model_prefix, model_with_head_base)

        model_base.add_adapter("dummy")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_base.save_adapter(temp_dir, "dummy")

            loading_info = {}
            model_with_head.load_adapter(temp_dir, loading_info=loading_info)

        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # check equal output
        input_data = self.get_input_samples((1, 128), config=model_with_head.config)
        output1 = model_with_head(**input_data)
        output2 = model_base(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))
