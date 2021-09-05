import unittest

import torch

from transformers import AutoModelWithHeads, BertConfig, BertForSequenceClassification
from transformers.adapters.composition import BatchSplit, Fuse, Parallel, Split, Stack, parse_composition
from transformers.testing_utils import require_torch, torch_device

from .test_modeling_common import ids_tensor


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

    def batched_training_pass(self):
        inputs = {"input_ids": ids_tensor((4, 128), 1000), "labels": torch.ones(4, dtype=torch.long)}
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
        self.model.add_adapter_fusion(Fuse("b", "d"))

        # fuse two stacks
        self.model.set_active_adapters(Fuse(Stack("a", "b"), Stack("c", "d")))

        self.training_pass()

    def test_mixed_stack(self):
        self.model.add_adapter_fusion(Fuse("a", "b"))

        self.model.set_active_adapters(Stack("a", Split("c", "d", split_index=64), Fuse("a", "b")))

        self.training_pass()

    def test_nested_split(self):
        # split into two stacks
        self.model.set_active_adapters(Split(Split("a", "b", split_index=32), "c", split_index=64))

        self.training_pass()

    def test_parallel(self):
        self.model.set_active_adapters(Parallel("a", "b", "c", "d"))

        inputs = {}
        inputs["input_ids"] = ids_tensor((1, 128), 1000)
        logits = self.model(**inputs).logits
        self.assertEqual(logits.shape, (4, 2))

    def test_nested_parallel(self):
        self.model.set_active_adapters(Stack("a", Parallel(Stack("b", "c"), "d")))

        inputs = {}
        inputs["input_ids"] = ids_tensor((1, 128), 1000)
        logits = self.model(**inputs).logits
        self.assertEqual(logits.shape, (2, 2))

    def test_batch_split(self):
        self.model.set_active_adapters(BatchSplit("a", "b", "c", batch_sizes=[1, 1, 2]))
        self.batched_training_pass()

    def test_batch_split_int(self):
        self.model.set_active_adapters(BatchSplit("a", "b", batch_sizes=2))
        self.batched_training_pass()

    def test_nested_batch_split(self):
        self.model.set_active_adapters(Stack("a", BatchSplit("b", "c", batch_sizes=[2, 2])))
        self.batched_training_pass()

    def test_batch_split_invalid(self):
        self.model.set_active_adapters(BatchSplit("a", "b", batch_sizes=[3, 4]))
        with self.assertRaises(IndexError):
            self.batched_training_pass()

    def test_batch_split_equivalent(self):
        self.model.set_active_adapters("a")
        self.model.eval()
        input_ids = ids_tensor((2, 128), 1000)
        output_a = self.model(input_ids[:1])

        self.model.set_active_adapters("b")
        output_b = self.model(input_ids[1:2])

        self.model.set_active_adapters(BatchSplit("a", "b", batch_sizes=[1, 1]))
        output = self.model(input_ids)

        self.assertTrue(torch.allclose(output_a[0], output[0][0], atol=1e-6))
        self.assertTrue(torch.allclose(output_b[0], output[0][1], atol=1e-6))


@require_torch
class ParallelAdapterInferenceTestMixin:
    def test_parallel_inference_with_heads(self):
        model = AutoModelWithHeads.from_config(self.config())

        model.add_adapter("a")
        model.add_adapter("b")
        model.add_classification_head("a", num_labels=2)
        model.add_classification_head("b", num_labels=3)
        model.eval()

        inputs = self.get_input_samples((2, 128), config=model.config)
        inputs["attention_mask"] = torch.randint(0, 2, size=(2, 128))

        # for reference, pass through single adapters
        model.active_adapters = "a"
        model.active_head = "a"
        outputs_a = model(**inputs)
        model.active_adapters = "b"
        model.active_head = "b"
        outputs_b = model(**inputs)

        model.active_adapters = Parallel("a", "b")
        # active_adapters should set parallel heads too
        self.assertEqual(model.active_head, ["a", "b"])
        outputs = model(**inputs)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0][0].shape, (2, 2))
        self.assertEqual(outputs[1][0].shape, (2, 3))
        self.assertTrue(torch.allclose(outputs[0][0], outputs_a[0]))
        self.assertTrue(torch.allclose(outputs[1][0], outputs_b[0]))

    def test_parallel_inference_with_wrong_number_of_heads(self):
        model = AutoModelWithHeads.from_config(self.config())
        model.eval()

        model.add_adapter("a")
        model.add_adapter("b")
        model.add_classification_head("a", num_labels=2)

        inputs = self.get_input_samples((2, 128), config=model.config)

        model.active_adapters = Parallel("a", "b")
        model.active_head = ["a"]
        with self.assertRaises(ValueError):
            model(**inputs)

        model.active_head = "a"
        with self.assertRaises(ValueError):
            model(**inputs)

    def test_batch_split_with_heads(self):
        model = AutoModelWithHeads.from_config(self.config())
        model.add_adapter("a")
        model.add_adapter("b")
        model.add_classification_head("a", num_labels=2)
        model.add_classification_head("b", num_labels=3)
        model.eval()

        inputs = self.get_input_samples((2, 128), config=model.config)["input_ids"]

        # for reference, pass through single adapters
        model.active_adapters = "a"
        model.active_head = "a"
        outputs_a = model(inputs[:1])
        model.active_adapters = "b"
        model.active_head = "b"
        outputs_b = model(inputs[1:])

        model.set_active_adapters(BatchSplit("a", "b", batch_sizes=[1, 1]))
        output = model(inputs)

        self.assertEqual(2, len(output))
        self.assertTrue(torch.allclose(output[0]["logits"], outputs_a["logits"]))
        self.assertTrue(torch.allclose(output[1]["logits"], outputs_b["logits"]))
