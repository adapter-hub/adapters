import copy
import random

import torch

from adapters import ADAPTER_MODEL_MAPPING, AutoAdapterModel, PrefixTuningConfig, SeqBnConfig, T5AdapterModel
from adapters.composition import BatchSplit, Parallel
from adapters.models.bert_generation.adapter_model import BertGenerationAdapterModel
from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, Trainer, TrainingArguments
from transformers.testing_utils import require_torch, torch_device


def filter_parameters(model, filter_string):
    return {k: v for (k, v) in model.named_parameters() if filter_string in k}


@require_torch
class ParallelAdapterInferenceTestMixin:
    def test_parallel_inference_with_heads(self):
        model = AutoAdapterModel.from_config(self.config())

        model.add_adapter("a")
        model.add_adapter("b")
        self.add_head(model, "a", num_labels=2)
        self.add_head(model, "b", num_labels=3)
        model.eval()
        model.to(torch_device)

        inputs = self.get_input_samples(config=model.config)
        inputs["attention_mask"] = torch.randint(0, 2, size=(3, 64), device=torch_device)

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
        if self.config_class in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING:
            self.assertEqual(outputs[0][0].shape, (3, 2))
            self.assertEqual(outputs[1][0].shape, (3, 3))
        self.assertTrue(torch.allclose(outputs[0][0], outputs_a[0], atol=1e-5))
        self.assertTrue(torch.allclose(outputs[1][0], outputs_b[0], atol=1e-5))

    def test_parallel_inference_with_wrong_number_of_heads(self):
        model = AutoAdapterModel.from_config(self.config())
        model.eval()

        model.add_adapter("a")
        model.add_adapter("b")
        self.add_head(model, "a", num_labels=2)
        model.to(torch_device)

        inputs = self.get_input_samples(config=model.config)

        model.active_adapters = Parallel("a", "b")
        model.active_head = ["a"]
        with self.assertRaises(ValueError):
            model(**inputs)

        model.active_head = "a"
        with self.assertRaises(ValueError):
            model(**inputs)

    def test_batch_split_with_heads(self):
        model = AutoAdapterModel.from_config(self.config())
        model.add_adapter("a")
        model.add_adapter("b")
        self.add_head(model, "a", num_labels=2)
        self.add_head(model, "b", num_labels=3)
        model.eval()
        model.to(torch_device)

        inputs = self.get_input_samples(config=model.config)
        if isinstance(model, T5AdapterModel):
            inputs["decoder_input_ids"] = inputs["input_ids"]

        # for reference, pass through single adapters
        model.active_adapters = "a"
        model.active_head = "a"
        outputs_a = model(**{k: v[:1] for k, v in inputs.items()})
        model.active_adapters = "b"
        model.active_head = "b"
        outputs_b = model(**{k: v[1:] for k, v in inputs.items()})

        model.set_active_adapters(BatchSplit("a", "b", batch_sizes=[1, 2]))
        output = model(**inputs)

        self.assertEqual(2, len(output))
        self.assertTrue(
            torch.allclose(
                output[0]["logits"],
                outputs_a["logits"],
                atol=1e-05,
            )
        )
        self.assertTrue(
            torch.allclose(
                output[1]["logits"],
                outputs_b["logits"],
                atol=1e-05,
            )
        )

    def test_parallel_generate(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING or (
            "seq2seq_lm" not in ADAPTER_MODEL_MAPPING[self.config_class].head_types
            and "causal_lm" not in ADAPTER_MODEL_MAPPING[self.config_class].head_types
        ):
            self.skipTest("No seq2seq or causal language model head")

        model1 = AutoAdapterModel.from_config(self.config())
        model1.add_adapter("adapter1")
        model1.add_adapter("adapter2")
        if "seq2seq_lm" in ADAPTER_MODEL_MAPPING[self.config_class].head_types:
            model1.add_seq2seq_lm_head("adapter1")
            model1.add_seq2seq_lm_head("adapter2")
        else:
            model1.add_causal_lm_head("adapter1")
            model1.add_causal_lm_head("adapter2")
        model1.set_active_adapters(Parallel("adapter1", "adapter2"))
        model1.to(torch_device)

        seq_output_length = 32

        # Finally, also check if generation works properly
        input_ids = self.get_input_samples((1, 4), config=model1.config)["input_ids"]
        input_ids = input_ids.to(torch_device)
        generated = model1.generate(input_ids, max_length=seq_output_length)
        self.assertLessEqual(generated.shape, (2, seq_output_length))


class ParallelTrainingMixin:
    def create_twin_adapters(self, model, name, adapter_config):
        # create adapter
        adapter1, adapter2 = name + "_1", name + "_2"
        model.add_adapter(adapter1, config=adapter_config)
        self.add_head(model, adapter1)
        # create a twin initialized with the same random weights
        model.add_adapter(adapter2, config=adapter_config)
        self.add_head(model, adapter2)

        state_dict = model.state_dict()
        for k, v in state_dict.items():
            if adapter1 in k:
                state_dict[k.replace(adapter1, adapter2)] = v
        model.load_state_dict(state_dict)

        return adapter1, adapter2

    def train_model(self, model, dataset):
        # trains model in eval mode for 2 epochs
        random.seed(42)
        torch.manual_seed(42)
        # Depending on the used optimizer the adapters are not exactly the same
        model.to(torch_device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for epoch in range(2):
            for data_input in dataset:
                for key, value in data_input.items():
                    data_input[key] = value.to(torch_device)

                optimizer.zero_grad()
                output = model(**data_input)
                loss = output["loss"]
                loss.backward()
                optimizer.step()
        return model

    def run_parallel_training_test(self, adapter_config, filter_key):
        model = AutoAdapterModel.from_config(self.config())

        model.add_adapter("mrpc1", config=adapter_config)
        model.add_adapter("mrpc2", config=adapter_config)
        self.add_head(model, "mrpc1")
        self.add_head(model, "mrpc2")
        model.active_adapters = Parallel("mrpc1", "mrpc2")
        model.train_adapter(Parallel("mrpc1", "mrpc2"))
        # model.eval()

        # all weights of the adapter should be activated
        for k, v in filter_parameters(model, filter_key.format("mrpc1")).items():
            self.assertTrue(v.requires_grad, k)
        # all weights of the adapter not used for training should be frozen
        for k, v in filter_parameters(model, filter_key.format("mrpc1")).items():
            self.assertTrue(v.requires_grad, k)
        # weights of the model should be frozen (check on some examples)
        for k, v in filter_parameters(model, "encoder.layer.0.attention").items():
            if filter_key.format("mrpc1") not in k and filter_key.format("mrpc2") not in k:
                self.assertFalse(v.requires_grad, k)

        state_dict_pre = copy.deepcopy(model.state_dict())

        train_dataset = self.dataset()
        training_args = TrainingArguments(
            output_dir="./examples",
            do_train=True,
            learning_rate=1.0,
            max_steps=20,
            no_cuda=True,
            remove_unused_columns=False,
        )

        # evaluate
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

        # check that the weights of the adapters have changed
        self.assertTrue(
            any([not torch.equal(v, state_dict_pre[k]) for k, v in model.state_dict().items() if "mrpc" in k])
        )
        self.assertTrue(
            all(torch.equal(v, state_dict_pre[k]) for k, v in model.state_dict().items() if "mrpc" not in k)
        )

    def run_parallel_training_equivalent_to_single(self, adapter_config):
        model = AutoAdapterModel.from_config(self.config())
        model.eval()

        a1, a2 = self.create_twin_adapters(model, "a", adapter_config)
        b1, b2 = self.create_twin_adapters(model, "b", adapter_config)

        dataset = []
        for i in range(3):
            input_data = self.get_input_samples(config=model.config)
            if isinstance(model, BertGenerationAdapterModel):
                input_data["labels"] = torch.randint(0, 2, (3, 64))
            else:
                input_data["labels"] = torch.randint(0, 2, (3, 1))
            dataset.append(input_data)

        for adapter in [a1, b1]:
            model.active_head = adapter
            model.set_active_adapters(adapter)
            model.train_adapter(adapter)
            model.eval()

            model = self.train_model(model, dataset)

        model.set_active_adapters(Parallel(a2, b2))
        model.train_adapter((Parallel(a2, b2)))
        model.eval()

        model = self.train_model(model, dataset)

        state_dict = model.state_dict()
        for k, v in state_dict.items():
            if a1 in k:
                self.assertTrue(
                    torch.allclose(v, state_dict[k.replace(a1, a2)], atol=1e-5),
                    torch.max(torch.sub(v, state_dict[k.replace(a1, a2)])),
                )
            if b1 in k:
                self.assertTrue(torch.allclose(v, state_dict[k.replace(b1, b2)], atol=1e-5))

    def test_parallel_training_bottleneck(self):
        self.run_parallel_training_test(SeqBnConfig(), "adapters.{}")

    def test_parallel_training_prefix_tuning(self):
        self.run_parallel_training_test(PrefixTuningConfig(), "prefix_tunings.{}")

    def test_parallel_training_equivalent_to_single_bottleneck(self):
        self.run_parallel_training_equivalent_to_single(SeqBnConfig())

    def test_parallel_training_equivalent_to_single_prefix_tuning(self):
        self.run_parallel_training_equivalent_to_single(PrefixTuningConfig())

    def test_parallel_training_single_forward_pass(self):
        model = AutoAdapterModel.from_config(self.config())
        model.eval()

        a1, a2 = self.create_twin_adapters(model, "a", SeqBnConfig())
        b1, b2 = self.create_twin_adapters(model, "b", SeqBnConfig())

        state_dict = model.state_dict()
        for k, v in state_dict.items():
            if a1 in k:
                self.assertTrue(torch.equal(v, state_dict[k.replace(a1, a2)]))
            if b1 in k:
                self.assertTrue(torch.equal(v, state_dict[k.replace(b1, b2)]))

        input_data = self.get_input_samples(config=model.config)
        if isinstance(model, BertGenerationAdapterModel):
            input_data["labels"] = torch.randint(0, 2, (3, 64), device=torch_device)
        else:
            input_data["labels"] = torch.randint(0, 2, (3, 1), device=torch_device)

        outputs = []
        for adapter in [a1, b1]:
            model.active_head = adapter
            model.set_active_adapters(adapter)
            model.train_adapter(adapter)
            model.eval()
            model.to(torch_device)
            outputs.append(model(**input_data))

        model.set_active_adapters(Parallel(a2, b2))
        model.train_adapter((Parallel(a2, b2)))
        model.eval()
        model.to(torch_device)

        parallel_outputs = model(**input_data)

        for out1, out2 in zip(outputs, parallel_outputs.head_outputs):
            self.assertTrue(torch.allclose(out1["loss"], out2["loss"]))
            self.assertTrue(torch.allclose(out1["logits"], out2["logits"], atol=1e-5))
