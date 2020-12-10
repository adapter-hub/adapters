import tempfile
import unittest

import torch

from tests.test_modeling_common import ids_tensor
from transformers import AutoConfig, AutoModelWithHeads
from transformers.adapter_heads import PredictionHead


class CustomHead(PredictionHead):
    def __init__(self, name, config, model):
        super().__init__(name)
        self.config = config
        self.build(model=model)

    def forward(self, outputs, attention_mask, return_dict, **kwargs):
        logits = super().forward(outputs[0])
        outputs = (logits,) + outputs[2:]
        return outputs


class AdapterCustomHeadTest(unittest.TestCase):
    def test_add_custom_head(self):
        model_name = "bert-base-uncased"
        model = AutoModelWithHeads.from_pretrained(model_name)
        model.register_custom_head("tag", CustomHead)
        config = {"head_type": "tag", "num_labels": 3, "layers": 2, "activation_function": "tanh"}
        model.add_custom_head("custom_head", config)
        model.eval()
        in_data = ids_tensor((1, 128), 1000)
        output1 = model(in_data)
        model.add_tagging_head("tagging_head", num_labels=3, layers=2)
        output2 = model(in_data)
        self.assertEqual(output1[0].size(), output2[0].size())

    def test_custom_head_from_model_config(self):
        model_name = "bert-base-uncased"
        model_config = AutoConfig.from_pretrained(model_name, custom_heads={"tag": CustomHead})
        model = AutoModelWithHeads.from_pretrained(model_name, config=model_config)
        config = {"head_type": "tag", "num_labels": 3, "layers": 2, "activation_function": "tanh"}
        model.add_custom_head("custom_head", config)
        model.eval()
        in_data = ids_tensor((1, 128), 1000)
        output1 = model(in_data)
        model.add_tagging_head("tagging_head", num_labels=3, layers=2)
        output2 = model(in_data)
        self.assertEqual(output1[0].size(), output2[0].size())

    def test_save_load_custom_head(self):
        model_name = "bert-base-uncased"
        model_config = AutoConfig.from_pretrained(model_name, custom_heads={"tag": CustomHead})
        model1 = AutoModelWithHeads.from_pretrained(model_name, config=model_config)
        model2 = AutoModelWithHeads.from_pretrained(model_name, config=model_config)
        config = {"head_type": "tag", "num_labels": 3, "layers": 2, "activation_function": "tanh"}
        model1.add_custom_head("custom_head", config)

        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_head(temp_dir, "custom_head")
            model2.load_head(temp_dir)

        model1.eval()
        model2.eval()

        in_data = ids_tensor((1, 128), 1000)
        output1 = model1(in_data)
        output2 = model2(in_data)
        self.assertEqual(output1[0].size(), output2[0].size())
        state1 = model1.heads["custom_head"].state_dict()
        state2 = model2.heads["custom_head"].state_dict()
        for ((k1, v1), (k2, v2)) in zip(state1.items(), state2.items()):
            self.assertTrue(torch.equal(v1, v2))
