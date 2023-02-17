import unittest

import torch

from tests.test_modeling_common import ids_tensor
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PfeifferConfig,
    PrefixTuningConfig,
)
from transformers.adapters.composition import BatchSplit, Fuse, Parallel, Split, Stack, parse_composition
from transformers.testing_utils import require_torch, torch_device


class AdapterCompositionParsingTest(unittest.TestCase):
    def test_parse_lists(self):
        self.assertEqual(Stack("a"), parse_composition("a"))
        self.assertEqual(Stack("a", "b", "c"), parse_composition(["a", "b", "c"]))
        self.assertEqual(Stack("a", Fuse("b", "c")), parse_composition(["a", ["b", "c"]]))

    def test_to_deep(self):
        self.assertRaises(ValueError, lambda: parse_composition(Stack("a", Fuse("b", Stack(Fuse("c", "d"), "e")))))

    def test_invalid_nesting_fusion(self):
        self.assertRaises(ValueError, lambda: parse_composition(Fuse(Fuse("a", "b"), "c")))
        self.assertRaises(ValueError, lambda: parse_composition(Fuse(Split("a", "b", 128), "c")))

    def test_invalid_nesting_split(self):
        self.assertRaises(ValueError, lambda: parse_composition(Split("a", Fuse("b", "c"), 128)))


@require_torch
class AdapterCompositionTest(unittest.TestCase):
    unsupported_blocks = []

    def get_adapter_config(self):
        return PfeifferConfig()

    def build_model(self):
        model = BertForSequenceClassification(BertConfig())
        adapter_config = self.get_adapter_config()
        model.add_adapter("a", config=adapter_config)
        model.add_adapter("b", config=adapter_config)
        model.add_adapter("c", config=adapter_config)
        model.add_adapter("d", config=adapter_config)
        model.to(torch_device)
        model.train()

        return model

    def training_pass(self, model):
        inputs = {}
        inputs["input_ids"] = ids_tensor((1, 128), 1000).to(torch_device)
        inputs["labels"] = torch.ones(1, dtype=torch.long).to(torch_device)
        loss = model(**inputs).loss
        loss.backward()

    def batched_training_pass(self, model):
        inputs = {
            "input_ids": ids_tensor((4, 128), 1000).to(torch_device),
            "labels": torch.ones(4, dtype=torch.long).to(torch_device),
        }
        loss = model(**inputs).loss
        loss.backward()

    def test_simple_stack(self):
        if Stack in self.unsupported_blocks:
            self.skipTest("Stack not supported by adapter config.")

        model = self.build_model()
        model.set_active_adapters(Stack("a", "b", "c", "d"))
        self.training_pass(model)

    def test_simple_split(self):
        if Split in self.unsupported_blocks:
            self.skipTest("Split not supported by adapter config.")

        model = self.build_model()
        # pass over split setup
        model.set_active_adapters(Split("a", "b", 64))

        self.training_pass(model)

    def test_stacked_split(self):
        if Stack in self.unsupported_blocks or Split in self.unsupported_blocks:
            self.skipTest("Stack or Split not supported by adapter config.")

        model = self.build_model()
        # split into two stacks
        model.set_active_adapters(Split(Stack("a", "b"), Stack("c", "d"), split_index=64))

        self.training_pass(model)

    def test_stacked_fusion(self):
        if Stack in self.unsupported_blocks or Fuse in self.unsupported_blocks:
            self.skipTest("Stack or Fuse not supported by adapter config.")

        model = self.build_model()
        model.add_adapter_fusion(Fuse("b", "d"))
        model.to(torch_device)

        # fuse two stacks
        model.set_active_adapters(Fuse(Stack("a", "b"), Stack("c", "d")))

        self.training_pass(model)

    def test_mixed_stack(self):
        if Stack in self.unsupported_blocks or Fuse in self.unsupported_blocks:
            self.skipTest("Stack or Fuse not supported by adapter config.")

        model = self.build_model()
        model.add_adapter_fusion(Fuse("a", "b"))
        model.to(torch_device)

        model.set_active_adapters(Stack("a", Split("c", "d", split_index=64), Fuse("a", "b")))

        self.training_pass(model)

    def test_nested_split(self):
        if Split in self.unsupported_blocks:
            self.skipTest("Split not supported by adapter config.")

        model = self.build_model()
        # split into two stacks
        model.set_active_adapters(Split(Split("a", "b", split_index=32), "c", split_index=64))

        self.training_pass(model)

    def test_parallel(self):
        if Parallel in self.unsupported_blocks:
            self.skipTest("Parallel not supported by adapter config.")

        model = self.build_model()
        model.set_active_adapters(Parallel("a", "b", "c", "d"))

        inputs = {}
        inputs["input_ids"] = ids_tensor((1, 128), 1000)
        logits = model(**inputs).logits
        self.assertEqual(logits.shape, (4, 2))

    def test_nested_parallel(self):
        if Parallel in self.unsupported_blocks or Stack in self.unsupported_blocks:
            self.skipTest("Parallel or Stack not supported by adapter config.")

        model = self.build_model()
        model.set_active_adapters(Stack("a", Parallel(Stack("b", "c"), "d")))

        inputs = {}
        inputs["input_ids"] = ids_tensor((1, 128), 1000)
        logits = model(**inputs).logits
        self.assertEqual(logits.shape, (2, 2))

    def test_batch_split(self):
        if BatchSplit in self.unsupported_blocks:
            self.skipTest("BatchSplit not supported by adapter config.")

        model = self.build_model()
        model.set_active_adapters(BatchSplit("a", "b", "c", batch_sizes=[1, 1, 2]))
        self.batched_training_pass(model)

    def test_batch_split_int(self):
        if BatchSplit in self.unsupported_blocks:
            self.skipTest("BatchSplit not supported by adapter config.")

        model = self.build_model()
        model.set_active_adapters(BatchSplit("a", "b", batch_sizes=2))
        self.batched_training_pass(model)

    def test_nested_batch_split_1(self):
        if BatchSplit in self.unsupported_blocks or Stack in self.unsupported_blocks:
            self.skipTest("BatchSplit or Stack not supported by adapter config.")

        model = self.build_model()
        model.set_active_adapters(Stack("a", BatchSplit("b", "c", batch_sizes=[2, 2])))
        self.batched_training_pass(model)

    def test_nested_batch_split_2(self):
        if BatchSplit in self.unsupported_blocks or Stack in self.unsupported_blocks:
            self.skipTest("BatchSplit or Stack not supported by adapter config.")

        model = self.build_model()
        model.set_active_adapters(BatchSplit(Stack("a", "b"), "c", batch_sizes=[2, 2]))
        self.batched_training_pass(model)

    def test_batch_split_invalid(self):
        if BatchSplit in self.unsupported_blocks:
            self.skipTest("BatchSplit not supported by adapter config.")

        model = self.build_model()
        model.set_active_adapters(BatchSplit("a", "b", batch_sizes=[3, 4]))
        with self.assertRaises(IndexError):
            self.batched_training_pass(model)

    def test_batch_split_equivalent(self):
        if BatchSplit in self.unsupported_blocks:
            self.skipTest("BatchSplit not supported by adapter config.")

        model = self.build_model()
        model.set_active_adapters("a")
        model.eval()
        input_ids = ids_tensor((2, 128), 1000)
        output_a = model(input_ids[:1])

        model.set_active_adapters("b")
        output_b = model(input_ids[1:2])

        model.set_active_adapters(BatchSplit("a", "b", batch_sizes=[1, 1]))
        output = model(input_ids)

        self.assertTrue(torch.allclose(output_a[0], output[0][0], atol=1e-6))
        self.assertTrue(torch.allclose(output_b[0], output[0][1], atol=1e-6))


class PrefixTuningCompositionTest(AdapterCompositionTest):
    unsupported_blocks = [Split, Fuse]

    def get_adapter_config(self):
        return PrefixTuningConfig()
