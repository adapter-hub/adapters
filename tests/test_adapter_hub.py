import os
import unittest

from transformers import ADAPTER_CONFIG_MAP, AdapterConfig, BertForSequenceClassification, get_adapter_config_hash
from transformers.adapter_utils import find_in_index

from .test_modeling_common import ids_tensor
from .utils import require_torch


SAMPLE_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/hub-index.sample.json")


@require_torch
class AdapterHubTest(unittest.TestCase):
    search_samples = [
        ("t@ukp", "pfeiffer", "path/to/pfeiffer/ukp"),
        ("s@ukp", "pfeiffer", "path/to/pfeiffer/ukp"),
        ("xyz", "pfeiffer", None),
        ("t/s", None, "path/to/default"),
        ("t/s@ukp", "pfeiffer", "path/to/pfeiffer/ukp"),
        ("t/s", "pfeiffer", "path/to/pfeiffer/default"),
        ("t/s", "houlsby", "path/to/houlsby/example-org"),
    ]

    def test_find_in_index(self):
        for sample in self.search_samples:
            with self.subTest(sample=sample):
                config = ADAPTER_CONFIG_MAP[sample[1]] if sample[1] else None
                found_entry = find_in_index(sample[0], None, None, config, index_file=SAMPLE_INDEX)
                self.assertEqual(sample[2], found_entry)

    def test_load_adapter_from_hub(self):
        for config in ["pfeiffer", "houlsby"]:
            with self.subTest(config=config):
                model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

                loading_info = {}
                adapter_name = model.load_adapter(
                    "sts/mrpc@ukp", config=config, version="1", loading_info=loading_info
                )

                self.assertEqual(0, len(loading_info["missing_keys"]))

                # hotfix for unnecessary weights in old adapters
                unexpected_keys = [k for k in loading_info["unexpected_keys"] if "adapter_attention" not in k]
                self.assertEqual(0, len(unexpected_keys))

                self.assertIn(adapter_name, model.config.adapters.adapters)
                # check if config is valid
                expected_hash = get_adapter_config_hash(AdapterConfig.load(config))
                real_hash = get_adapter_config_hash(model.config.adapters.get(adapter_name))
                self.assertEqual(expected_hash, real_hash)

                # check size of output
                in_data = ids_tensor((1, 128), 1000)
                output = model(in_data)
                self.assertEqual([1, 2], list(output[0].size()))
