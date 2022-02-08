import unittest
from threading import Thread

from tests.test_modeling_common import ids_tensor
from transformers import AdapterSetup, AutoModelWithHeads, BertConfig
from transformers.testing_utils import require_torch, torch_device


@require_torch
class AdapterSetupContextTest(unittest.TestCase):
    def setUp(self):
        self.config = BertConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )

    def test_context_nested(self):
        model = AutoModelWithHeads.from_config(self.config)
        model.add_adapter("a")
        model.add_classification_head("a", num_labels=2)
        model.add_adapter("b")
        model.add_classification_head("b", num_labels=3)
        # Make sure no adapter is activated
        model.active_adapters = None
        model.active_head = None
        model.to(torch_device)
        in_data = {"input_ids": ids_tensor((1, 128), 1000)}

        # Set a hook before the adapter to make sure it's actually called.
        calls_a = 0
        calls_b = 0

        def forward_pre_hook_a(module, input):
            nonlocal calls_a
            calls_a += 1

        def forward_pre_hook_b(module, input):
            nonlocal calls_b
            calls_b += 1

        adapter_a = model.get_adapter("a")[0]["output_adapter"]
        adapter_a.register_forward_pre_hook(forward_pre_hook_a)
        adapter_b = model.get_adapter("b")[0]["output_adapter"]
        adapter_b.register_forward_pre_hook(forward_pre_hook_b)

        with AdapterSetup("a"):
            out_a = model(**in_data)

            with AdapterSetup("b"):
                out_b = model(**in_data)

        self.assertEqual(out_a[0].shape, (1, 2))
        self.assertEqual(calls_a, 1)
        self.assertEqual(out_b[0].shape, (1, 3))
        self.assertEqual(calls_b, 1)

    def test_context_multi_threading(self):
        model = AutoModelWithHeads.from_config(self.config)
        model.add_adapter("a")
        model.add_classification_head("a", num_labels=2)
        model.add_adapter("b")
        model.add_classification_head("b", num_labels=3)
        model.active_head = None
        model.to(torch_device)
        in_data = {"input_ids": ids_tensor((1, 128), 1000)}
        outputs = []
        hook_called = []

        def run_forward_pass(adapter_setup, expected_shape):
            calls = 0

            def forward_pre_hook(module, input):
                nonlocal calls
                calls += 1

            adapter = model.get_adapter(adapter_setup)[0]["output_adapter"]
            adapter.register_forward_pre_hook(forward_pre_hook)

            with AdapterSetup(adapter_setup):
                out = model(**in_data)
                outputs.append((out, expected_shape))
                hook_called.append(calls > 0)

        t1 = Thread(target=run_forward_pass, args=("a", (1, 2)))
        t2 = Thread(target=run_forward_pass, args=("b", (1, 3)))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(hook_called), 2)
        for out, expected_shape in outputs:
            self.assertEqual(out[0].shape, expected_shape)
        self.assertTrue(all(hook_called))
