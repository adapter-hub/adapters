import unittest

from transformers import AdapterConfig, BertForSequenceClassification, get_adapter_config_hash

from .test_modeling_common import ids_tensor
from .utils import require_torch


@require_torch
class AdapterHubTest(unittest.TestCase):
    # TODO add tests for default resolving when supported by Hub index

    def test_load_adapter_from_hub(self):
        for config in ["pfeiffer", "houlsby"]:
            with self.subTest(config=config):
                model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

                loading_info = {}
                adapter_name = model.load_adapter(
                    "sts/mrpc@ukp", config=config, version="1", loading_info=loading_info
                )

                self.assertEqual(0, len(loading_info["missing_keys"]))
                self.assertEqual(0, len(loading_info["unexpected_keys"]))

                self.assertIn(adapter_name, model.config.adapters.adapters)
                # check if config is valid
                expected_hash = get_adapter_config_hash(AdapterConfig.load(config))
                real_hash = get_adapter_config_hash(model.config.adapters.get(adapter_name))
                self.assertEqual(expected_hash, real_hash)

                # check size of output
                in_data = ids_tensor((1, 128), 1000)
                output = model(in_data)
                self.assertEqual([1, 2], list(output[0].size()))
