import unittest

from transformers import ADAPTER_CONFIG_MAP, AdapterType, BertModel, RobertaModel, XLMRobertaModel

from .utils import require_torch


@require_torch
class AdapterModelTest(unittest.TestCase):
    model_classes = [BertModel, RobertaModel, XLMRobertaModel]

    def test_model_config_serialization(self):
        """PretrainedConfigurations should not raise an Exception when serializing the config dict

        See, e.g., PretrainedConfig.to_json_string()
        """
        for model_class in self.model_classes:
            for k, v in ADAPTER_CONFIG_MAP.items():
                model_config = model_class.config_class
                model = model_class(model_config())
                model.add_adapter("test", adapter_type=AdapterType.text_task, config=v)
                # should not raise an exception
                model.config.to_json_string()
