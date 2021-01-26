import unittest

import torch

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModelWithHeads,
    DistilBertModelWithHeads,
    RobertaModelWithHeads,
)
from transformers.adapter_composition import Fuse, Parallel, Split, Stack, parse_composition
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


@require_torch
class ParallelAdapterInferenceTest(unittest.TestCase):

    model_classes = [BertModelWithHeads, RobertaModelWithHeads, DistilBertModelWithHeads]

    def test_parallel_inference_with_heads(self):
        for model_class in self.model_classes:
            model_config = model_class.config_class
            model = model_class(model_config())
            model.eval()

            model.add_adapter("a")
            model.add_adapter("b")
            model.add_classification_head("a", num_labels=2)
            model.add_classification_head("b", num_labels=3)

            with self.subTest(model_class=model_class.__name__):

                inputs = {}
                inputs["input_ids"] = ids_tensor((2, 128), 1000)

                # for reference, pass through single adapters
                model.set_active_adapters("a")
                model.set_active_heads("a")
                outputs_a = model(**inputs, return_dict=False)
                model.set_active_adapters("b")
                model.set_active_heads("b")
                outputs_b = model(**inputs, return_dict=False)

                model.set_active_adapters(Parallel("a", "b"))
                model.set_active_heads(["a", "b"])
                # TODO currently only works with return_dict=False
                outputs = model(**inputs, return_dict=False)

                self.assertEqual(len(outputs), 2)
                self.assertEqual(outputs[0][0].shape, (2, 2))
                self.assertEqual(outputs[1][0].shape, (2, 3))
                self.assertTrue(torch.equal(outputs[0][0], outputs_a[0]))
                self.assertTrue(torch.equal(outputs[1][0], outputs_b[0]))
