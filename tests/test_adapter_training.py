import copy
import unittest

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelWithHeads,
    AutoTokenizer,
    GlueDataset,
    GlueDataTrainingArguments,
    Trainer,
    TrainingArguments,
)
from transformers.testing_utils import require_torch


def filter_parameters(model, filter_string):
    return {k: v for (k, v) in model.named_parameters() if filter_string in k}


@require_torch
class AdapterTrainingTest(unittest.TestCase):

    model_names = ["bert-base-uncased", "distilbert-base-uncased"]

    def test_train_single_adapter(self):
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelWithHeads.from_pretrained(model_name)

                # add two adapters: one will be trained and the other should be frozen
                model.add_adapter("mrpc", "text_task")
                model.add_adapter("dummy", "text_task")
                model.add_classification_head("mrpc")

                self.assertIn("mrpc", model.config.adapters.adapters)
                self.assertIn("dummy", model.config.adapters.adapters)

                # train the mrpc adapter -> should be activated & unfreezed
                model.train_adapter("mrpc")
                self.assertEqual([["mrpc"]], model.active_adapters)

                # all weights of the adapter should be activated
                for k, v in filter_parameters(model, "text_task_adapters.mrpc").items():
                    self.assertTrue(v.requires_grad, k)
                # all weights of the adapter not used for training should be freezed
                for k, v in filter_parameters(model, "text_task_adapters.dummy").items():
                    self.assertFalse(v.requires_grad, k)
                # weights of the model should be freezed (check on some examples)
                for k, v in filter_parameters(model, "encoder.layer.0.attention").items():
                    self.assertFalse(v.requires_grad, k)

                state_dict_pre = copy.deepcopy(model.state_dict())

                # setup dataset
                data_args = GlueDataTrainingArguments(
                    task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
                )
                train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
                training_args = TrainingArguments(
                    output_dir="./examples", do_train=True, learning_rate=0.1, max_steps=5, no_cuda=True
                )

                # evaluate
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                )
                trainer.train()

                for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items()):
                    if "mrpc" in k1:
                        self.assertFalse(torch.equal(v1, v2))
                    else:
                        self.assertTrue(torch.equal(v1, v2))

    def test_train_adapter_fusion(self):
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)

                # load the adapters to be fused
                model.load_adapter("sts/mrpc@ukp", with_head=False)
                model.load_adapter("sts/qqp@ukp", with_head=False)
                model.load_adapter("sts/sts-b@ukp", with_head=False)

                self.assertIn("mrpc", model.config.adapters.adapters)
                self.assertIn("qqp", model.config.adapters.adapters)
                self.assertIn("sts-b", model.config.adapters.adapters)

                # setup fusion
                adapter_setup = [["mrpc", "qqp", "sts-b"]]
                model.add_fusion(adapter_setup[0])
                model.train_fusion(adapter_setup[0])
                model.set_active_adapters(adapter_setup)
                self.assertEqual(adapter_setup, model.active_adapters)

                # all weights of the adapters should be frozen (test for one)
                for k, v in filter_parameters(model, "text_task_adapters.mrpc").items():
                    self.assertFalse(v.requires_grad, k)
                # all weights of the fusion layer should be activated
                for k, v in filter_parameters(model, "adapter_fusion_layer").items():
                    self.assertTrue(v.requires_grad, k)
                # weights of the model should be freezed (check on some examples)
                for k, v in filter_parameters(model, "encoder.layer.0.attention").items():
                    self.assertFalse(v.requires_grad, k)

                state_dict_pre = copy.deepcopy(model.state_dict())

                # setup dataset
                data_args = GlueDataTrainingArguments(
                    task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
                )
                train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
                training_args = TrainingArguments(
                    output_dir="./examples", do_train=True, learning_rate=0.1, max_steps=5, no_cuda=True
                )

                # evaluate
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                )
                trainer.train()

                for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items()):
                    if "adapter_fusion_layer" in k1 or "classifier" in k1:
                        self.assertFalse(torch.equal(v1, v2), k1)
                    else:
                        self.assertTrue(torch.equal(v1, v2), k1)
