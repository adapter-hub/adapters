import copy
import tempfile

import torch

from transformers import AutoAdapterModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.testing_utils import require_torch, torch_device


def filter_parameters(model, filter_string):
    return {k: v for (k, v) in model.named_parameters() if filter_string in k}


@require_torch
class EmbeddingTestMixin:
    def test_load_embeddings(self):
        model = self.get_model()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_embeddings(tmp_dir, "default")
            model.load_embeddings(tmp_dir, "test")

        self.assertEqual(model.active_embeddings, "test")

    def test_add_embeddings(self):
        model = self.get_model()
        tokenizer = AutoTokenizer.from_pretrained("tests_adapters/fixtures/SiBERT")
        model.add_embeddings("test", tokenizer)
        self.assertEqual(model.active_embeddings, "test")

    def test_add_embedding_tokens(self):
        model = self.get_model()
        tokenizer = AutoTokenizer.from_pretrained("tests_adapters/fixtures/SiBERT")
        self.assertEqual(tokenizer.vocab_size, 10000)
        tokenizer.add_tokens(["test_token"])
        model.add_embeddings("test", tokenizer)
        self.assertEqual(model.get_input_embeddings().num_embeddings, 10001)

    def test_delete_embeddings(self):
        model = self.get_model()
        tokenizer = AutoTokenizer.from_pretrained("tests_adapters/fixtures/SiBERT")
        model.add_embeddings("test", tokenizer)
        self.assertEqual(model.active_embeddings, "test")
        model.delete_embeddings("test")
        self.assertFalse("test" in model.loaded_embeddings)
        self.assertEqual(model.active_embeddings, "default")

    def test_save_load_embedding(self):
        model = self.get_model()
        tokenizer = AutoTokenizer.from_pretrained("tests_adapters/fixtures/SiBERT")
        input_data = self.get_input_samples((1, 128), vocab_size=tokenizer.vocab_size, config=model.config)
        model.add_embeddings("test", tokenizer)
        model.eval()
        model.to(torch_device)
        output1 = model(**input_data)
        self.assertEqual(model.active_embeddings, "test")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_embeddings(tmp_dir, "test", tokenizer=tokenizer)
            tokenizer_ref = model.load_embeddings(tmp_dir, "test_reloaded")

        self.assertEqual(model.active_embeddings, "test_reloaded")
        model.to(torch_device)
        output2 = model(**input_data)
        self.assertTrue(
            torch.equal(model.loaded_embeddings["test"].weight, model.loaded_embeddings["test_reloaded"].weight)
        )
        self.assertTrue(torch.equal(output1[0], output2[0]))
        self.assertEqual(tokenizer.vocab, tokenizer_ref.vocab)

    def test_back_to_default(self):
        model = self.get_model()
        model.eval()
        input_data = self.get_input_samples((1, 128), config=model.config)
        output1 = model(**input_data)
        tokenizer = AutoTokenizer.from_pretrained("tests_adapters/fixtures/SiBERT")
        model.add_embeddings("test", tokenizer)
        self.assertEqual(model.active_embeddings, "test")
        model.set_active_embeddings("default")
        output2 = model(**input_data)
        self.assertEqual(model.active_embeddings, "default")
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_training_embedding(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoAdapterModel.from_config(self.config())
        model.add_embeddings("test", tokenizer)
        self.assertEqual(model.active_embeddings, "test")
        model.add_adapter("test")
        self.add_head(model, "test")
        model.train_adapter("test", train_embeddings=True)

        for k, v in filter_parameters(model, "adapters.test.").items():
            self.assertTrue(v.requires_grad, k)

        self.assertTrue(model.get_input_embeddings().train)

        state_dict_pre = copy.deepcopy(model.state_dict())

        train_dataset = self.dataset(tokenizer=tokenizer)
        training_args = TrainingArguments(
            output_dir="./examples",
            do_train=True,
            learning_rate=0.4,
            max_steps=15,
            no_cuda=True,
            per_device_train_batch_size=2,
        )

        # evaluate
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

        self.assertFalse(
            all(
                torch.equal(v1, v2)
                for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items())
                if "test" in k1
            )
        )
        self.assertTrue(
            all(
                torch.equal(v1, v2)
                for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items())
                if "test" not in k1
            )
        )

    def test_reference_embedding(self):
        model = AutoAdapterModel.from_config(self.config())  # self.get_model()
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        new_tokenizer = AutoTokenizer.from_pretrained("tests_adapters/fixtures/SiBERT")

        model.add_embeddings("test", new_tokenizer, "default", tokenizer)

        default_embedding = model.base_model.loaded_embeddings["default"]
        test_embedding = model.base_model.loaded_embeddings["test"]

        input_test = []
        input_default = []

        for v, idx in new_tokenizer.get_vocab().items():
            if v in tokenizer.get_vocab() and not v.startswith("##"):
                input_test.append(idx)
                input_default.append(tokenizer.get_vocab()[v])
                if len(input_test) >= 5:
                    break

        input_default = torch.tensor([input_default])
        input_test = torch.tensor([input_test])

        default = default_embedding(input_default)
        test = test_embedding(input_test)

        self.assertTrue(torch.equal(default, test))

        # activate for training
        model.add_adapter("test")
        model.train_adapter("test", train_embeddings=True)
