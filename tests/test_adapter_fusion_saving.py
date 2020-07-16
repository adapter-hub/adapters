import unittest

from transformers import ADAPTERFUSION_CONFIG_MAP, AdapterType, BertModel, RobertaModel, XLMRobertaModel

from .utils import require_torch


@require_torch
class AdapterFusionModelTest(unittest.TestCase):
    model_classes = [BertModel, RobertaModel, XLMRobertaModel]

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
