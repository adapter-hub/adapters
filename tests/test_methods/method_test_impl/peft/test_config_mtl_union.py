import copy
from dataclasses import asdict

from datasets import Dataset

from accelerate.state import torch
from accelerate.utils.modeling import tempfile
from adapters.composition import MultiTask
from adapters.configuration.adapter_config import MTLLoRAConfig, MultiTaskConfigUnion
from adapters.heads.language_modeling import CausalLMHead
from adapters.models.auto.adapter_model import ADAPTER_MODEL_MAPPING, AutoAdapterModel
from adapters.trainer import AdapterTrainer
from adapters.utils import WEIGHTS_NAME
from huggingface_hub import os
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from tests.test_methods.method_test_impl.utils import create_twin_models
from transformers.testing_utils import require_torch, torch_device
from transformers.training_args import TrainingArguments


class AdapterMethodMultiTaskConfigUnionTestMixin(AdapterMethodBaseTestMixin):
    def _set_filter_keys(self, filter_keys, task_names):
        return {fk.format(name="{name}", task=task) for fk in filter_keys for task in task_names}

    def run_load_test(self, adapter_config, **kwargs):
        model1, model2 = create_twin_models(self.model_class, self.config)

        name = "dummy_adapter"
        model1.add_adapter(name, config=adapter_config)
        model1.set_active_adapters(name)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            model1.save_adapter(temp_dir, name)

            # Check that there are actually weights saved
            # empty string is for union shared shared params.
            for adapter_name in ["", *adapter_config.task_names]:
                weights = torch.load(
                    os.path.join(temp_dir, adapter_name, WEIGHTS_NAME),
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertTrue(len(weights) > 0)

            # also tests that set_active works
            loading_info = {}
            model2.load_adapter(temp_dir, set_active=True, loading_info=loading_info)

        # check if all weights were loaded
        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # check if adapter was correctly loaded
        for adapter_name in [name, *adapter_config.task_names]:
            self.assertTrue(adapter_name in model2.adapters_config)

        # check equal output
        input_data = self.get_input_samples(config=model1.config, n_tasks=len(adapter_config.task_names))
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0], output2[0], atol=1e-4))

    def get_dataset_with_task_ids(self, tasks):
        train_dataset = self.get_dataset()
        if not isinstance(train_dataset, Dataset):
            train_dataset = Dataset.from_list([asdict(feature) for feature in train_dataset])
        else:
            # to get dataset with transformations
            train_dataset = Dataset.from_list([example for example in train_dataset])

        def add_task_ids(example_batch):
            inputs = copy.deepcopy(example_batch)
            bsz = len(example_batch)
            inputs["task_ids"] = torch.randint(0, len(tasks), (bsz,)).tolist()
            return inputs

        train_dataset.set_transform(add_task_ids)
        return train_dataset

    def trainings_run(
        self,
        model,
        lr=1.0,
        steps=8,
        batch_size=2,
        gradient_accumulation_steps=1,
        **kwargs,
    ):
        # setup dataset
        train_dataset = self.get_dataset_with_task_ids(kwargs["tasks"])

        training_args = TrainingArguments(
            output_dir="./examples",
            do_train=True,
            learning_rate=lr,
            max_steps=steps,
            use_cpu=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            remove_unused_columns=False,
        )

        # evaluate
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def run_train_test(self, adapter_config, filter_keys, tasks):
        if not self.do_run_train_tests:
            self.skipTest("Skipping training tests. Set `do_run_train_tests=True` to run them.")
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")

        def format_filter_keys(filter_keys, name, task_names):
            return {k.format(name=name, task=task) for k in filter_keys for task in task_names}

        model = AutoAdapterModel.from_config(self.config())

        name = adapter_config.__class__.__name__
        # add two adapters: one will be trained and the other should be frozen
        model.add_adapter(name, config=adapter_config)
        task_names = adapter_config.task_names

        dummy_task_names = [f"dummy_{t}" for t in task_names]
        dummy_name, dummy_config = "dummy", adapter_config.replace(task_names=dummy_task_names)

        model.add_adapter(dummy_name, config=dummy_config)
        self.add_head(model, name)
        # filter_keys.append("heads.{name}.")

        self._assert_adapter_available(model, name)
        self._assert_adapter_available(model, dummy_name)

        model.train_adapter(name)
        self.assertEqual(set(task_names), model.active_adapters.flatten())

        # all weights of the adapter should be activated
        has_weights = False
        filter_keys_trained = format_filter_keys(filter_keys, name=name, task_names=task_names)
        for k, v in self._filter_parameters(model, filter_keys_trained).items():
            has_weights = True
            self.assertTrue(v.requires_grad, k)
        self.assertTrue(has_weights)
        # all weights of the adapter not used for training should be frozen
        filter_keys_untrained = format_filter_keys(filter_keys, name=dummy_name, task_names=dummy_task_names)
        for k, v in self._filter_parameters(model, filter_keys_untrained).items():
            self.assertFalse(v.requires_grad, k)

        state_dict_pre = copy.deepcopy(model.state_dict())

        self.trainings_run(model, tasks=tasks)

        # check that the adapters have changed, but the base model has not
        adapters_with_change, base_with_change = False, False
        # check whether the key corresponds to a tied embedding

        def has_tied_embeddings(k):
            tied_embeddings = hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings
            is_tied_layer = (
                isinstance(model.heads[name], CausalLMHead)
                and "heads.{}.{}.weight".format(name, len(model.heads[name]._modules) - 1) in k
            )
            return tied_embeddings and is_tied_layer

        for (k1, v1), (k2, v2) in zip(state_dict_pre.items(), model.state_dict().items()):
            # move both to the same device to avoid device mismatch errors
            v1, v2 = v1.to(v2.device), v2
            if (any(key in k1 for key in filter_keys_trained) or name in k1) and not has_tied_embeddings(k1):
                adapters_with_change |= not torch.equal(v1, v2)
            else:
                base_with_change |= not torch.equal(v1, v2)

        self.assertTrue(adapters_with_change)
        self.assertFalse(base_with_change)


@require_torch
class MultiTaskConfigUnionAdapterTest(AdapterMethodMultiTaskConfigUnionTestMixin):

    adapter_configs_to_test = [
        (
            MultiTaskConfigUnion(
                base_config=MTLLoRAConfig(n_up_projection=3, init_weights="bert"),
                task_names=["a", "b", "c"],
            ),
            [
                ".shared_parameters.{name}.",
                ".loras.{task}.",
            ],
        ),
    ]

    def test_add_mtl_union_adapters(self):
        model = self.get_model()
        model.eval()
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                self.run_add_test(
                    model,
                    adapter_config,
                    self._set_filter_keys(filter_keys, adapter_config.task_names),
                )

    def test_add_mtl_union_adapters_with_set_active(self):
        model = self.get_model()
        model.eval()
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                name = f"test_adapter_{adapter_config.__class__.__name__}"
                model.add_adapter(name, config=adapter_config, set_active=True)
                model.set_active_adapters == MultiTask(*adapter_config.task_names)
                model.to(torch_device)

                # Adapter is correctly added to config
                self.assertTrue(name in model.adapters_config)
                self.assertEqual(adapter_config, model.adapters_config.get(name))

                # Check that weights are available and active
                has_weights = False
                filter_keys = self._set_filter_keys(filter_keys, adapter_config.task_names)
                filter_keys = [k.format(name=name) for k in filter_keys]
                for k, v in self._filter_parameters(model, filter_keys).items():
                    has_weights = True
                    self.assertTrue(v.requires_grad, k)
                self.assertTrue(has_weights)

                # Remove added adapters in case of multiple subtests
                model.set_active_adapters(None)
                model.delete_adapter(name)

    def test_delete_mtl_union_adapters(self):
        model = self.get_model()
        model.eval()
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                filter_keys = self._set_filter_keys(filter_keys, adapter_config.task_names)
                self.run_delete_test(model, adapter_config, filter_keys)

    def test_load_mtl_union_adapters(self):
        for adapter_config, _ in self.adapter_configs_to_test:
            with self.subTest(
                model_class=self.model_class.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                self.run_load_test(adapter_config, tasks=adapter_config.task_names)

    def test_mtl_union_adapter_forward(self):
        model = self.get_model()
        model.eval()
        for adapter_config, _ in self.adapter_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                self.run_forward_test(
                    model,
                    adapter_config,
                    tasks=adapter_config.task_names,
                    adapter_setup=MultiTask(*adapter_config.task_names),
                    n_tasks=len(adapter_config.task_names),
                )

    def test_mtl_union_adapter_train(self):
        model = self.get_model()
        model.eval()
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                self.run_train_test(
                    adapter_config,
                    filter_keys,
                    tasks=adapter_config.task_names,
                )
