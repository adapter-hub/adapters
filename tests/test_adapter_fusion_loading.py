import copy
import tempfile
import unittest

import torch

from transformers import (
    ADAPTER_CONFIG_MAP,
    ADAPTERFUSION_CONFIG_MAP,
    AdapterType,
    BertModel,
    RobertaModel,
    XLMRobertaModel,
)

from .test_modeling_common import ids_tensor
from .utils import require_torch


def create_twin_models(model1):
    model1.eval()
    # create a twin initialized with the same random weights
    model2 = copy.deepcopy(model1)
    model2.eval()
    return model1, model2


@require_torch
class AdapterFusionModelTest(unittest.TestCase):

    model_classes = [BertModel, RobertaModel, XLMRobertaModel]

    def test_add_adapter_fusion(self):

        for adater_fusion_config_name, adapter_fusion_config in ADAPTERFUSION_CONFIG_MAP.items():
            for config_name, adapter_config in ADAPTER_CONFIG_MAP.items():
                for type_name, adapter_type in AdapterType.__members__.items():
                    for model_class in self.model_classes:
                        model_config = model_class.config_class
                        model = model_class(model_config())

                        # skip configs without invertible language adapters
                        if adapter_type == AdapterType.text_lang and not adapter_config.invertible_adapter:
                            continue
                        with self.subTest(model_class=model_class, config=config_name, adapter_type=type_name):
                            name1 = f"{type_name}-{config_name}-1"
                            name2 = f"{type_name}-{config_name}-2"
                            model.add_adapter(name1, adapter_type, config=adapter_config)
                            model.add_adapter(name2, adapter_type, config=adapter_config)

                            # adapter is correctly added to config
                            self.assertTrue(name1 in model.config.adapters.adapter_list(adapter_type))
                            self.assertTrue(name2 in model.config.adapters.adapter_list(adapter_type))
                            self.assertEqual(adapter_config, model.config.adapters.get(name1))
                            self.assertEqual(adapter_config, model.config.adapters.get(name2))

                            model.add_fusion([name1, name2], adater_fusion_config_name)

                            # check forward pass
                            input_ids = ids_tensor((1, 128), 1000)
                            input_data = {"input_ids": input_ids}
                            if adapter_type == AdapterType.text_task or adapter_type == AdapterType.text_lang:
                                input_data["adapter_names"] = [[name1, name2]]
                            adapter_output = model(**input_data)
                            base_output = model(input_ids)
                            self.assertEqual(len(adapter_output), len(base_output))
                            self.assertFalse(torch.equal(adapter_output[0], base_output[0]))

    def test_load_adapter_fusion(self):
        for adater_fusion_config_name, adapter_fusion_config in ADAPTERFUSION_CONFIG_MAP.items():
            for name, adapter_type in AdapterType.__members__.items():
                for model_class in self.model_classes:
                    with self.subTest(model_class=model_class, adapter_type=name):
                        model_config = model_class.config_class
                        model1 = model_class(model_config())
                        name1 = "name1"
                        name2 = "name2"
                        model1.add_adapter(name1, adapter_type)
                        model1.add_adapter(name2, adapter_type)
                        model1, model2 = create_twin_models(model1)

                        model1.add_fusion([name1, name2], adater_fusion_config_name)
                        with tempfile.TemporaryDirectory() as temp_dir:
                            model1.save_adapter_fusion(temp_dir, ",".join([name1, name2]))
                            model2.load_adapter_fusion(temp_dir)

                        model1.eval()
                        model2.eval()
                        # check if adapter was correctly loaded
                        self.assertTrue(model1.config.adapter_fusion_models == model2.config.adapter_fusion_models)

                        # check equal output
                        in_data = ids_tensor((1, 128), 1000)
                        output1 = model1(in_data, adapter_names=[[name1, name2]])
                        output2 = model2(in_data, adapter_names=[[name1, name2]])
                        self.assertEqual(len(output1), len(output2))
                        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_model_config_serialization(self):
        """PretrainedConfigurations should not raise an Exception when serializing the config dict

        See, e.g., PretrainedConfig.to_json_string()
        """
        for model_class in self.model_classes:
            for k, v in ADAPTERFUSION_CONFIG_MAP.items():
                model_config = model_class.config_class
                model = model_class(model_config())
                model.add_adapter("test1", AdapterType.text_task)
                model.add_adapter("test2", AdapterType.text_task)
                model.add_fusion(["test1", "test2"], adapter_fusion_config=v)
                # should not raise an exception
                model.config.to_json_string()
