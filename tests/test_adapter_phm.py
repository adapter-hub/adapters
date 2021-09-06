import copy
import random
import unittest

import torch
from torch import Tensor

from tests.test_adapter_training import filter_parameters
from transformers import (
    AutoModel,
    AutoModelWithHeads,
    AutoTokenizer,
    GlueDataset,
    GlueDataTrainingArguments,
    Trainer,
    TrainingArguments,
)
from transformers.adapters.configuration import CompactorConfig, PfeifferCompactConfig


def shared_parameters(param: Tensor):
    def getter():
        return param

    return getter


class TestSaveLabel(unittest.TestCase):
    def get_input_samples(self, shape, vocab_size=5000, config=None):
        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(random.randint(0, vocab_size - 1))
        input_ids = torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()
        # this is needed e.g. for BART
        if config and config.eos_token_id is not None:
            input_ids[input_ids == config.eos_token_id] = random.randint(0, config.eos_token_id - 1)
            input_ids[:, -1] = config.eos_token_id

        return input_ids

    def test_model(self):
        name = "compactor"
        compactor_config = CompactorConfig(phm_dim=16)
        adapter_config = PfeifferCompactConfig(compactor=compactor_config)
        model = AutoModel.from_pretrained("bert-base-uncased")
        model.add_adapter(name, config=adapter_config)
        self.assertTrue(name in model.config.adapters)
        self.assertEqual(adapter_config, model.config.adapters.get(name))

        model.train_adapter(name)
        self.assertEqual(set([name]), model.active_adapters.flatten())
        for k, v in dict(model.named_parameters()).items():
            if name in k:
                self.assertTrue(v.requires_grad)
            else:
                self.assertFalse(v.requires_grad)

        input_data = self.get_input_samples((1, 128), config=model.config)
        output_data = model(input_data)
        self.assertTrue((1, 128, 768), output_data["last_hidden_state"].shape)

    def test_training(self):
        model_name = "bert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelWithHeads.from_pretrained(model_name)

        # add two adapters: one will be trained and the other should be frozen

        compactor_config = CompactorConfig(phm_dim=16)
        adapter_config = PfeifferCompactConfig(compactor=compactor_config)
        model.add_adapter("mrpc", config=adapter_config)
        model.add_classification_head("mrpc")

        self.assertIn("mrpc", model.config.adapters.adapters)

        # train the mrpc adapter -> should be activated & unfreezed
        model.train_adapter("mrpc")
        self.assertEqual(set(["mrpc"]), model.active_adapters.flatten())

        # all weights of the adapter should be activated
        for k, v in filter_parameters(model, "mrpc.").items():
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
            output_dir="./examples", do_train=True, learning_rate=0.1, max_steps=7, no_cuda=True
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
