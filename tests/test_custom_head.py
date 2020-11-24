import unittest

import torch as torch

from tests.test_modeling_common import ids_tensor
from transformers.adapter_bert import PredictionHead
from transformers import AutoModelWithHeads


class CustomHead(PredictionHead):
    def __init__(self, name, config, model):
        super().__init__(name)
        self.config = config
        self.build(model=model)

    def forward(self, outputs, attention_mask, labels, return_dict):
        logits = self.head(outputs[0])
        outputs = (logits,) + outputs[2:]
        return outputs


class MyTestCase(unittest.TestCase):
    def test_add_costom_head(self):
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
