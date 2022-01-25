import copy

import torch

from transformers import AutoModelWithHeads, AdapterConfig, PfeifferCompactorConfig, TrainingArguments, AdapterTrainer, \
    AutoTokenizer
from tests.test_modeling_common import ids_tensor
from transformers.testing_utils import require_torch


def filter_parameters(model, filter_string):
    return {k: v for (k, v) in model.named_parameters() if filter_string in k}

@require_torch
class CompacterTestMixin:
    def trainings_run(self, model, tokenizer):
        # setup dataset
        train_dataset = self.dataset(tokenizer)
        training_args = TrainingArguments(
            output_dir="./examples",
            do_train=True,
            learning_rate=0.7,
            max_steps=20,
            no_cuda=True,
            per_device_train_batch_size=2,
        )

        # evaluate
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def test_add_compacter(self):
        model = AutoModelWithHeads.from_config(self.config())
        adapter_config = PfeifferCompactorConfig()

        # add two adapters: one will be trained and the other should be frozen
        model.add_adapter("compacter", config=adapter_config)

        self.assertIn("compacter", model.config.adapters.adapters)

        # train the mrpc adapter -> should be activated & unfreezed
        model.train_adapter("compacter")
        self.assertEqual(set(["compacter"]), model.active_adapters.flatten())

        # all weights of the adapter should be activated
        for k, v in filter_parameters(model, "adapters.compacter.").items():
            self.assertTrue(v.requires_grad, k)
        # weights of the model should be freezed (check on some examples)
        for k, v in filter_parameters(model, "encoder.layer.0.attention").items():
            self.assertFalse(v.requires_grad, k)

    def test_forward_compacter(self):
        model = AutoModelWithHeads.from_config(self.config())
        adapter_config = PfeifferCompactorConfig()

        model.add_adapter("compacter", config=adapter_config)
        model.add_classification_head("compacter", num_labels=3)
        model.set_active_adapters("compacter")
        self.assertEqual(set(["compacter"]), model.active_adapters.flatten())
        input_tensor = ids_tensor((1, 128), vocab_size=1000)
        output = model(input_tensor)
        self.assertEqual((1, 3), output["logits"].shape)

    def test_shared_phm_compacter(self):
        model = AutoModelWithHeads.from_config(self.config())
        adapter_config = PfeifferCompactorConfig(shared_W_phm=True)

        model.add_adapter("compacter", config=adapter_config)
        model.add_classification_head("compacter", num_labels=3)

        model.set_active_adapters("compacter")

        input_tensor = ids_tensor((1, 128), vocab_size=1000)
        output = model(input_tensor)
        self.assertEqual((1, 3), output["logits"].shape)

    def test_train_shared_W_compacter(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        model = AutoModelWithHeads.from_config(self.config())
        adapter_config = PfeifferCompactorConfig(shared_W_phm=True, shared_phm_rule=False)

        model.add_adapter("compacter", config=adapter_config)
        model.add_classification_head("compacter", num_labels=3)

        model.train_adapter("compacter")

        parameters_pre = copy.deepcopy(model.base_model.shared_parameters)
        self.trainings_run(model, tokenizer)

        self.assertTrue(
            any(any(not torch.equal(p1, p2) for p1, p2 in zip(p_pre.values(), p_post.values())) for p_pre, p_post in zip(parameters_pre.values(), model.base_model.shared_parameters.values()))
        )

    def test_train_shared_phm_compacter(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        model = AutoModelWithHeads.from_config(self.config())
        adapter_config = PfeifferCompactorConfig()
        model.add_adapter("compacter", config=adapter_config)
        model.add_classification_head("compacter", num_labels=3)

        model.train_adapter("compacter")

        parameters_pre = copy.deepcopy(model.base_model.shared_parameters)
        self.trainings_run(model, tokenizer)

        self.assertTrue(
            any(any(not torch.equal(p1, p2) for p1, p2 in zip(p_pre.values(), p_post.values())) for p_pre, p_post in zip(parameters_pre.values(), model.base_model.shared_parameters.values()))
        )


