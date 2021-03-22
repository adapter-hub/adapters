import copy
import tempfile

import torch

from transformers import (
    ADAPTER_CONFIG_MAP,
    AutoModel,
    AutoModelWithHeads,
    HoulsbyConfig,
    HoulsbyInvConfig,
    PfeifferConfig,
    PfeifferInvConfig,
)
from transformers.testing_utils import require_torch

from .test_modeling_common import ids_tensor


def create_twin_models(model_class, config_creator=None):
    if config_creator:
        model_config = config_creator()
        model1 = model_class.from_config(model_config)
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
        model = AutoModel.from_config(self.config())
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
                input_ids = ids_tensor((1, 128), 1000)
                input_data = {"input_ids": input_ids}
                adapter_output = model(**input_data)
                model.set_active_adapters(None)
                base_output = model(**input_data)
                self.assertEqual(len(adapter_output), len(base_output))
                self.assertFalse(torch.equal(adapter_output[0], base_output[0]))

    def test_add_adapter_with_invertible(self):
        model = AutoModel.from_config(self.config())
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
                input_ids = ids_tensor((1, 128), 1000)
                input_data = {"input_ids": input_ids}
                adapter_output = model(**input_data)
                # make sure the output is different without invertible adapter
                del model.invertible_adapters[name]
                adapter_output_no_inv = model(**input_data)
                self.assertEqual(len(adapter_output), len(adapter_output_no_inv))
                self.assertFalse(torch.equal(adapter_output[0], adapter_output_no_inv[0]))

    def test_load_adapter(self):
        model1, model2 = create_twin_models(AutoModel, self.config)

        name = "dummy"
        model1.add_adapter(name)
        model1.set_active_adapters([name])
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter(temp_dir, name)

            model2.load_adapter(temp_dir)
            model2.set_active_adapters([name])

        # check if adapter was correctly loaded
        self.assertTrue(name in model2.config.adapters)

        # check equal output
        in_data = ids_tensor((1, 128), 1000)
        output1 = model1(in_data)
        output2 = model2(in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_load_full_model(self):
        model1 = AutoModel.from_config(self.config())
        model1.eval()

        name = "dummy"
        model1.add_adapter(name)
        model1.set_active_adapters([name])
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_pretrained(temp_dir)

            model2 = AutoModel.from_pretrained(temp_dir)
            model2.set_active_adapters([name])

        # check if adapter was correctly loaded
        self.assertTrue(name in model2.config.adapters)

        # check equal output
        in_data = ids_tensor((1, 128), 1000)
        output1 = model1(in_data)
        output2 = model2(in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_model_config_serialization(self):
        """PretrainedConfigurations should not raise an Exception when serializing the config dict

        See, e.g., PretrainedConfig.to_json_string()
        """
        for k, v in ADAPTER_CONFIG_MAP.items():
            model = AutoModel.from_config(self.config())
            model.add_adapter("test", config=v)
            # should not raise an exception
            model.config.to_json_string()

    def test_loading_adapter_weights_with_prefix(self):
        model_base, model_with_head_base = create_twin_models(AutoModel, self.config)

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
        in_data = ids_tensor((1, 128), 1000)
        # this is needed e.g. for BART
        if model_with_head.config.eos_token_id is not None:
            in_data[:, -1] = model_with_head.config.eos_token_id
        output1 = model_with_head(in_data)
        output2 = model_base(in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_loading_adapter_weights_without_prefix(self):
        model_base, model_with_head_base = create_twin_models(AutoModel, self.config)

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
        in_data = ids_tensor((1, 128), 1000)
        # this is needed e.g. for BART
        if model_with_head.config.eos_token_id is not None:
            in_data[:, -1] = model_with_head.config.eos_token_id
        output1 = model_with_head(in_data)
        output2 = model_base(in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))
