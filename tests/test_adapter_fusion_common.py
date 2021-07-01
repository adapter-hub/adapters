import copy
import tempfile
from dataclasses import asdict

import torch

from transformers import ADAPTERFUSION_CONFIG_MAP, AdapterConfig, AutoModel, PfeifferConfig
from transformers.testing_utils import require_torch


@require_torch
class AdapterFusionModelTestMixin:
    def test_add_adapter_fusion(self):
        config_name = "pfeiffer"
        adapter_config = AdapterConfig.load(config_name)

        for adater_fusion_config_name, adapter_fusion_config in ADAPTERFUSION_CONFIG_MAP.items():
            model = AutoModel.from_config(self.config())
            model.eval()

            with self.subTest(model_class=model.__class__.__name__, config=config_name):
                name1 = f"{config_name}-1"
                name2 = f"{config_name}-2"
                model.add_adapter(name1, config=config_name)
                model.add_adapter(name2, config=config_name)

                # adapter is correctly added to config
                self.assertTrue(name1 in model.config.adapters)
                self.assertTrue(name2 in model.config.adapters)
                self.assertEqual(asdict(adapter_config), asdict(model.config.adapters.get(name1)))
                self.assertEqual(asdict(adapter_config), asdict(model.config.adapters.get(name2)))

                model.add_adapter_fusion([name1, name2], adater_fusion_config_name)

                # check forward pass
                input_ids = self.get_input_samples((1, 128), config=model.config)
                input_data = {"input_ids": input_ids}
                model.set_active_adapters([[name1, name2]])
                adapter_output = model(**input_data)
                model.set_active_adapters(None)
                base_output = model(**input_data)
                self.assertEqual(len(adapter_output), len(base_output))
                self.assertFalse(torch.equal(adapter_output[0], base_output[0]))

    def test_add_adapter_fusion_different_config(self):
        model = AutoModel.from_config(self.config())
        model.eval()

        # fusion between a and b should be possible whereas fusion between a and c should fail
        model.add_adapter("a", config=PfeifferConfig(reduction_factor=16))
        model.add_adapter("b", config=PfeifferConfig(reduction_factor=2))
        model.add_adapter("c", config="houlsby")

        # correct fusion
        model.add_adapter_fusion(["a", "b"])
        self.assertIn("a,b", model.config.adapter_fusion_models)
        # failing fusion
        self.assertRaises(ValueError, lambda: model.add_adapter_fusion(["a", "c"]))

    def test_delete_adapter_fusion(self):
        model = AutoModel.from_config(self.config())
        model.eval()

        name1 = "test_adapter_1"
        name2 = "test_adapter_2"
        model.add_adapter(name1, config="houlsby")
        model.add_adapter(name2, config="houlsby")
        self.assertTrue(name1 in model.config.adapters)
        self.assertTrue(name2 in model.config.adapters)

        model.add_adapter_fusion([name1, name2])
        self.assertTrue(",".join([name1, name2]) in model.config.adapter_fusion_models)

        model.delete_adapter_fusion([name1, name2])
        self.assertFalse(",".join([name1, name2]) in model.config.adapter_fusion_models)

    def test_load_adapter_fusion(self):
        for adater_fusion_config_name, adapter_fusion_config in ADAPTERFUSION_CONFIG_MAP.items():
            model1 = AutoModel.from_config(self.config())
            model1.eval()

            with self.subTest(model_class=model1.__class__.__name__):
                name1 = "name1"
                name2 = "name2"
                model1.add_adapter(name1)
                model1.add_adapter(name2)

                model2 = copy.deepcopy(model1)
                model2.eval()

                model1.add_adapter_fusion([name1, name2], adater_fusion_config_name)
                with tempfile.TemporaryDirectory() as temp_dir:
                    model1.save_adapter_fusion(temp_dir, ",".join([name1, name2]))
                    model2.load_adapter_fusion(temp_dir)

                # check if adapter was correctly loaded
                self.assertTrue(model1.config.adapter_fusion_models == model2.config.adapter_fusion_models)

                # check equal output
                in_data = self.get_input_samples((1, 128), config=model1.config)
                model1.set_active_adapters([[name1, name2]])
                model2.set_active_adapters([[name1, name2]])
                output1 = model1(in_data)
                output2 = model2(in_data)
                self.assertEqual(len(output1), len(output2))
                self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_load_full_model_fusion(self):
        model1 = AutoModel.from_config(self.config())
        model1.eval()

        name1 = "name1"
        name2 = "name2"
        model1.add_adapter(name1)
        model1.add_adapter(name2)
        model1.add_adapter_fusion([name1, name2])
        # save & reload model
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_pretrained(temp_dir)
            model2 = AutoModel.from_pretrained(temp_dir)

        # check if AdapterFusion was correctly loaded
        self.assertTrue(model1.config.adapter_fusion_models == model2.config.adapter_fusion_models)

        # check equal output
        in_data = self.get_input_samples((1, 128), config=model1.config)
        model1.set_active_adapters([[name1, name2]])
        model2.set_active_adapters([[name1, name2]])
        output1 = model1(in_data)
        output2 = model2(in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_model_config_serialization_fusion(self):
        """PretrainedConfigurations should not raise an Exception when serializing the config dict

        See, e.g., PretrainedConfig.to_json_string()
        """
        for k, v in ADAPTERFUSION_CONFIG_MAP.items():
            model = AutoModel.from_config(self.config())
            model.add_adapter("test1")
            model.add_adapter("test2")
            model.add_adapter_fusion(["test1", "test2"], adapter_fusion_config=v)
            # should not raise an exception
            model.config.to_json_string()
