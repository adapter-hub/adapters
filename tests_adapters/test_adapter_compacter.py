import copy

import torch

from transformers import AutoAdapterModel, TrainingArguments, AdapterTrainer, \
    AutoTokenizer, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, ADAPTER_MODEL_MAPPING
from transformers.testing_utils import require_torch
from transformers.adapters.configuration import CompacterPlusPlusConfig


def filter_parameters(model, filter_string):
    return {k: v for (k, v) in model.named_parameters() if filter_string in k}


@require_torch
class CompacterTestMixin:
    def trainings_run(self, model, tokenizer):
        # setup dataset
        train_dataset = self.dataset(tokenizer)
        training_args = TrainingArguments(
            output_dir="../tests/examples",
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
        model = self.get_model()
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8)

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
        model = self.get_model()
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8)

        model.add_adapter("compacter", config=adapter_config)
        model.set_active_adapters("compacter")
        self.assertEqual(set(["compacter"]), model.active_adapters.flatten())
        input_tensor = self.get_input_samples((2, 128), config=model.config)
        output = model(**input_tensor)
        self.assertEqual(2, output[0].shape[0])

    def test_shared_phm_compacter(self):
        model = self.get_model()
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, shared_W_phm=True, reduction_factor=8)

        model.add_adapter("compacter", config=adapter_config)

        model.set_active_adapters("compacter")

        input_tensor = self.get_input_samples((2, 128), config=model.config)
        output = model(**input_tensor)
        self.assertEqual(2, output[0].shape[0])

    def test_train_shared_w_compacter(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoAdapterModel.from_config(self.config())
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, shared_W_phm=True, shared_phm_rule=False, reduction_factor=8)

        model.add_adapter("compacter", config=adapter_config)
        self.add_head(model, "compacter", num_labels=3)

        model.train_adapter("compacter")

        parameters_pre = copy.deepcopy(model.base_model.shared_parameters)
        self.trainings_run(model, tokenizer)

        self.assertTrue(
            any(any(not torch.equal(p1, p2) for p1, p2 in zip(p_pre.values(), p_post.values())) for p_pre, p_post in
                zip(parameters_pre.values(), model.base_model.shared_parameters.values()))
        )

    def test_train_shared_phm_compacter(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoAdapterModel.from_config(self.config())
        adapter_config = CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8)
        model.add_adapter("compacter", config=adapter_config)
        self.add_head(model, "compacter", num_labels=3)

        model.train_adapter("compacter")

        parameters_pre = copy.deepcopy(model.base_model.shared_parameters)
        self.trainings_run(model, tokenizer)

        self.assertTrue(
            any(any(not torch.equal(p1, p2) for p1, p2 in zip(p_pre.values(), p_post.values())) for p_pre, p_post in
                zip(parameters_pre.values(), model.base_model.shared_parameters.values()))
        )

