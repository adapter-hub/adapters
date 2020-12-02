import unittest

import torch

from transformers import BertConfig
from transformers.adapter_composition import Fuse, Split, Stack
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.testing_utils import require_torch, torch_device

from .test_modeling_common import ids_tensor


@require_torch
class AdapterCompositionTest(unittest.TestCase):
    def setUp(self):
        self.model = BertForSequenceClassification(BertConfig())
        self.model.add_adapter("a")
        self.model.add_adapter("b")
        self.model.add_adapter("c")
        self.model.add_adapter("d")
        self.model.to(torch_device)
        self.model.train()

    def training_pass(self):
        inputs = {}
        inputs["input_ids"] = ids_tensor((1, 128), 1000)
        inputs["labels"] = torch.ones(1, dtype=torch.long)
        loss = self.model(**inputs).loss
        loss.backward()

    def test_simple_split(self):
        # pass over split setup
        self.model.set_active_adapters(Split("a", "b", 64))

        self.training_pass()

    def test_stacked_split(self):
        # split into two stacks
        self.model.set_active_adapters(Split(Stack("a", "b"), Stack("c", "d"), split_index=64))

        self.training_pass()

    def test_stacked_fusion(self):
        self.model.add_fusion(Fuse("b", "d"))

        # fuse two stacks
        self.model.set_active_adapters(Fuse(Stack("a", "b"), Stack("c", "d")))

        self.training_pass()

    def test_mixed_stack(self):
        self.model.add_fusion(Fuse("a", "b"))

        self.model.set_active_adapters(Stack("a", Split("c", "d", split_index=64), Fuse("a", "b")))

        self.training_pass()
