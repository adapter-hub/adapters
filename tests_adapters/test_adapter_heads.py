import os
import tempfile

import torch

from transformers import ADAPTER_MODEL_MAPPING, AdapterSetup, AutoAdapterModel, AutoModelForSequenceClassification
from transformers.adapters.composition import BatchSplit, Stack
from transformers.testing_utils import require_torch, torch_device

from .methods import create_twin_models


@require_torch
class PredictionHeadModelTestMixin:

    batch_size = 1
    seq_length = 128

    def run_prediction_head_test(
        self, model, compare_model, head_name, input_shape=None, output_shape=(1, 2), label_dict=None
    ):
        # first, check if the head is actually correctly registered as part of the pt module
        self.assertTrue(f"heads.{head_name}" in dict(model.named_modules()))

        # save & reload
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_head(temp_dir, head_name)

            compare_model.load_head(temp_dir)

        # check if adapter was correctly loaded
        self.assertTrue(head_name in compare_model.heads)

        model.to(torch_device)
        compare_model.to(torch_device)

        # make a forward pass
        model.active_head = head_name
        input_shape = input_shape or (self.batch_size, self.seq_length)
        in_data = self.get_input_samples(input_shape, config=model.config)
        if label_dict:
            for k, v in label_dict.items():
                in_data[k] = v
        output1 = model(**in_data)
        # For the Seq2SeqLMOutput logits are at index 0
        # ToDo figure out why
        idx = "logits" if hasattr(output1, "logits") else 1
        self.assertEqual(output_shape, tuple(output1[idx].size()))
        # check equal output
        compare_model.active_head = head_name
        output2 = compare_model(**in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[idx], output2[idx]))

    def test_classification_head(self):
        if not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_classification_head"):
            self.skipTest("No classification head")

        model1, model2 = create_twin_models(AutoAdapterModel, self.config)

        model1.add_classification_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(model1, model2, "dummy", label_dict=label_dict)

    def test_image_classification_head(self):
        if not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_image_classification_head"):
            self.skipTest("No image classification head")

        model1, model2 = create_twin_models(AutoAdapterModel, self.config)

        model1.add_image_classification_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(model1, model2, "dummy", input_shape=(1, 3, 224, 224), label_dict=label_dict)

    def test_multiple_choice_head(self):
        if not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_multiple_choice_head"):
            self.skipTest("No multiple choice head")

        model1, model2 = create_twin_models(AutoAdapterModel, self.config)

        model1.add_multiple_choice_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.ones(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(
            model1, model2, "dummy", input_shape=(self.batch_size, 2, self.seq_length), label_dict=label_dict
        )

    def test_tagging_head(self):
        if not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_tagging_head"):
            self.skipTest("No tagging head")

        model1, model2 = create_twin_models(AutoAdapterModel, self.config)

        model1.add_tagging_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(
            model1, model2, "dummy", output_shape=(self.batch_size, self.seq_length, 2), label_dict=label_dict
        )

    def test_qa_head(self):
        if not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_qa_head"):
            self.skipTest("No QA head")

        model1, model2 = create_twin_models(AutoAdapterModel, self.config)

        model1.add_qa_head("dummy")
        label_dict = {}
        label_dict["start_positions"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        label_dict["end_positions"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(
            model1, model2, "dummy", output_shape=(self.batch_size, self.seq_length), label_dict=label_dict
        )

    def test_causal_lm_head(self):
        if not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_causal_lm_head"):
            self.skipTest("No causal language model head")

        model1, model2 = create_twin_models(AutoAdapterModel, self.config)
        model1.add_causal_lm_head("dummy")

        label_dict = {}
        # Use a different length for the seq2seq output
        seq_output_length = self.seq_length + 30
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)

        self.run_prediction_head_test(
            model1,
            model2,
            "dummy",
            output_shape=(self.batch_size, self.seq_length, model1.config.vocab_size),
            label_dict=label_dict,
        )

        # Finally, also check if generation works properly
        input_ids = self.get_input_samples((1, self.seq_length), config=model1.config)["input_ids"]
        input_ids = input_ids.to(torch_device)
        generated = model1.generate(input_ids, max_length=seq_output_length)
        self.assertEqual(generated.shape, (1, seq_output_length))

    def test_seq2seq_lm_head(self):
        if not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_seq2seq_lm_head"):
            self.skipTest("No seq2seq language model head")

        model1, model2 = create_twin_models(AutoAdapterModel, self.config)
        model1.add_seq2seq_lm_head("dummy")

        label_dict = {}
        # Use a different length for the seq2seq output
        seq_output_length = self.seq_length + 30
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)

        # prepare decoder_input_ids similar to how DataCollatorForSeq2Seq does it
        if hasattr(model1, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = model1.prepare_decoder_input_ids_from_labels(labels=label_dict["labels"])
            label_dict["decoder_input_ids"] = decoder_input_ids

        self.run_prediction_head_test(
            model1,
            model2,
            "dummy",
            output_shape=(self.batch_size, self.seq_length, model1.config.vocab_size),
            label_dict=label_dict,
        )

        # Finally, also check if generation works properly
        input_ids = self.get_input_samples((1, self.seq_length), config=model1.config)["input_ids"]
        input_ids = input_ids.to(torch_device)
        generated = model1.generate(input_ids, max_length=seq_output_length)
        self.assertEqual(generated.shape, (1, seq_output_length))

    def test_masked_lm_head(self):
        if not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_masked_lm_head"):
            self.skipTest("No causal or seq2seq language model head")

        model1, model2 = create_twin_models(AutoAdapterModel, self.config)

        model1.add_masked_lm_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(
            model1,
            model2,
            "dummy",
            output_shape=(self.batch_size, self.seq_length, model1.config.vocab_size),
            label_dict=label_dict,
        )

    def test_dependency_parsing_head(self):
        if not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_dependency_parsing_head"):
            self.skipTest("No dependency parsing head")

        model1, model2 = create_twin_models(AutoAdapterModel, self.config)

        model1.add_dependency_parsing_head("dummy")
        label_dict = {}
        label_dict["labels_arcs"] = torch.zeros(
            (self.batch_size, self.seq_length), dtype=torch.long, device=torch_device
        )
        label_dict["labels_rels"] = torch.zeros(
            (self.batch_size, self.seq_length), dtype=torch.long, device=torch_device
        )
        label_dict["word_starts"] = torch.zeros(
            (self.batch_size, self.seq_length), dtype=torch.long, device=torch_device
        )
        self.run_prediction_head_test(
            model1, model2, "dummy", output_shape=(1, self.seq_length, self.seq_length + 1, 2), label_dict=label_dict
        )

    def test_delete_head(self):
        model = AutoAdapterModel.from_config(self.config())
        model.eval()

        name = "test_head"
        self.add_head(model, name)
        self.assertTrue(name in model.heads)
        self.assertTrue(name in model.config.prediction_heads)
        self.assertEqual(name, model.active_head)

        model.delete_head(name)
        self.assertFalse(name in model.heads)
        self.assertFalse(name in model.config.prediction_heads)
        self.assertNotEqual(name, model.active_head)

    def test_adapter_with_head(self):
        model1, model2 = create_twin_models(AutoAdapterModel, self.config)

        name = "dummy"
        model1.add_adapter(name)
        output_size = self.add_head(model1, name, num_labels=3)
        model1.set_active_adapters(name)
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter(temp_dir, name)

            model2.load_adapter(temp_dir)
            model2.set_active_adapters(name)
        # check equal output
        in_data = self.get_input_samples(config=model1.config)
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**in_data)
        output2 = model2(**in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))
        self.assertEqual(output_size, output1[0].size()[1])

    def test_adapter_with_head_load_as(self):
        model1, model2 = create_twin_models(AutoAdapterModel, self.config)

        name = "dummy"
        model1.add_adapter(name)
        output_size = self.add_head(model1, name, num_labels=3)
        model1.set_active_adapters(name)
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter(temp_dir, name)

            # reload using a different name
            model2.load_adapter(temp_dir, load_as="new_name")
            model2.set_active_adapters("new_name")

        # check equal output
        in_data = self.get_input_samples(config=model1.config)
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**in_data)
        output2 = model2(**in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))
        self.assertEqual(output_size, output1[0].size()[1])

    def test_load_full_model(self):
        model = AutoAdapterModel.from_config(self.config())
        self.add_head(model, "dummy", layers=1)

        true_config = model.get_prediction_heads_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            # save
            model.save_pretrained(temp_dir)
            # reload
            model = AutoAdapterModel.from_pretrained(temp_dir)
        self.assertIn("dummy", model.heads)
        self.assertDictEqual(true_config, model.get_prediction_heads_config())

    def test_batch_split_head(self):
        model = AutoAdapterModel.from_config(self.config())
        output_size_a = self.add_head(model, "a", num_labels=2)
        output_size_b = self.add_head(model, "b", num_labels=2)
        model.active_head = BatchSplit("a", "b", batch_sizes=[1, 2])

        in_data = self.get_input_samples(config=model.config)
        model.to(torch_device)
        out = model(**in_data)

        self.assertEqual(2, len(out))
        self.assertEqual((1, output_size_a), out[0][0].shape[:2])
        self.assertEqual((2, output_size_b), out[1][0].shape[:2])

    def test_batch_split_adapter_head(self):
        model = AutoAdapterModel.from_config(self.config())
        self.add_head(model, "a")
        self.add_head(model, "b")
        model.add_adapter("a")
        model.add_adapter("b")
        model.add_adapter("c")
        model.set_active_adapters(BatchSplit(Stack("c", "a"), "b", batch_sizes=[2, 1]))

        in_data = self.get_input_samples(config=model.config)
        model.to(torch_device)
        out = model(**in_data)

        self.assertEqual(2, len(out))
        self.assertTrue(isinstance(model.active_head, BatchSplit))

    def test_reload_static_to_flex_head(self):
        if not hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_classification_head"):
            self.skipTest("No classification head available")
        static_head_model = AutoModelForSequenceClassification.from_config(self.config())
        flex_head_model = AutoAdapterModel.from_pretrained(
            None, config=self.config(), state_dict=static_head_model.state_dict()
        )
        static_head_model.eval()
        flex_head_model.eval()

        static_head_model.add_adapter("test")

        with tempfile.TemporaryDirectory() as temp_dir:
            static_head_model.save_adapter(temp_dir, "test")

            loading_info = {}
            flex_head_model.load_adapter(temp_dir, loading_info=loading_info)

            # Load the adapter a second time to make sure our conversion script doesn't break anything
            flex_head_model.load_adapter(temp_dir, loading_info=loading_info)
        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # adapter and head were loaded
        self.assertIn("test", flex_head_model.config.adapters)
        self.assertIn("test", flex_head_model.heads)

        # check equal output
        in_data = self.get_input_samples(config=flex_head_model.config)
        static_head_model.to(torch_device)
        flex_head_model.to(torch_device)
        with AdapterSetup("test"):
            output1 = static_head_model(**in_data)
            output2 = flex_head_model(**in_data, head="test")
        self.assertTrue(torch.all(torch.isclose(output1.logits, output2.logits)))

    def test_invertible_adapter_with_head(self):
        if hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_masked_lm_head"):
            lm_head = "masked_lm"
        elif hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_causal_lm_head"):
            lm_head = "casual_lm"
        elif hasattr(ADAPTER_MODEL_MAPPING[self.config_class], "add_seq2seq_lm_head"):
            lm_head = "seq2seq_lm"
        else:
            self.skipTest("No masked or causel language model head")

        model = AutoAdapterModel.from_config(self.config())
        model.add_adapter("test", config="pfeiffer+inv")
        if lm_head == "casual_lm":
            model.add_causal_lm_head("test")
        elif lm_head == "masked_lm":
            model.add_masked_lm_head("test")
        elif lm_head == "seq2seq_lm":
            model.add_seq2seq_lm_head("test")
        else:
            raise RuntimeError("{} is not a valid lm head".format(lm_head))
        model.set_active_adapters("test")

        # Set a hook before the invertible adapter to make sure it's actually called twice:
        # Once after the embedding layer and once in the prediction head.
        calls = 0

        def forward_pre_hook(module, input):
            nonlocal calls
            calls += 1

        inv_adapter = model.base_model.get_invertible_adapter()
        self.assertIsNotNone(inv_adapter)
        inv_adapter.register_forward_pre_hook(forward_pre_hook)

        in_data = self.get_input_samples((self.batch_size, self.seq_length), config=model.config)
        model.to(torch_device)
        out = model(**in_data)

        self.assertEqual((self.batch_size, self.seq_length, model.config.vocab_size), out[0].shape)
        self.assertEqual(2, calls)

    def test_context_simple(self):
        model = AutoAdapterModel.from_config(self.config())
        model.add_adapter("a")
        output_size = self.add_head(model, "a", num_labels=3)
        # Make sure no adapter is activated
        model.active_adapters = None
        model.active_head = None
        model.to(torch_device)
        in_data = self.get_input_samples(config=model.config)

        # Set a hook before the adapter to make sure it's actually called.
        calls = 0

        def forward_pre_hook(module, input):
            nonlocal calls
            calls += 1

        adapter = model.get_adapter("a")[0]["output_adapter"]
        adapter.register_forward_pre_hook(forward_pre_hook)

        with AdapterSetup("a"):
            out = model(**in_data)

        self.assertEqual(out[0].shape[:2], (3, output_size))
        self.assertEqual(calls, 1)

    def test_save_all_adapters_with_head(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")

        model = AutoAdapterModel.from_config(self.config())
        model.eval()
        model.add_adapter("test")
        self.add_head(model, "test")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_all_adapters(tmp_dir, with_head=True)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "test", "head_config.json")))

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_all_adapters(tmp_dir, with_head=False)
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, "test", "head_config.json")))
