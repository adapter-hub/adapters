from copy import deepcopy
from dataclasses import asdict

import torch
from datasets import Dataset

from adapters.composition import MultiTask
from adapters.configuration.adapter_config import LoRAConfig, MTLLoRAConfig
from adapters.context import AdapterSetup
from adapters.heads.language_modeling import CausalLMHead
from adapters.models.auto.adapter_model import ADAPTER_MODEL_MAPPING, AutoAdapterModel
from adapters.trainer import AdapterTrainer
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch, torch_device
from transformers.training_args import TrainingArguments


# torch.autograd.set_detect_anomaly(True)


class MultiTaskTestMethod(AdapterMethodBaseTestMixin):
    def get_dataset_with_task_ids(self, tasks):
        train_dataset = self.get_dataset()
        if not isinstance(train_dataset, Dataset):
            train_dataset = Dataset.from_list([asdict(feature) for feature in train_dataset])
        else:
            # to get dataset with transformations
            train_dataset = Dataset.from_list([example for example in train_dataset])

        def add_task_ids(example_batch):
            inputs = deepcopy(example_batch)
            bsz = len(example_batch)
            inputs["task_ids"] = torch.randint(0, len(tasks), (bsz,)).tolist()
            return inputs

        train_dataset.set_transform(add_task_ids)
        return train_dataset

    def _get_filter_keys(self, filter_keys, task_names, name):
        return {k.format(name=name, task=task) for k in filter_keys for task in task_names}

    def trainings_run(
        self,
        model,
        lr=1e-5,
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

    def run_train_test(
        self,
        adapter_config,
        filter_keys,
        adapter_setup,
        shared_parameters,
        tasks,
    ):
        if not self.do_run_train_tests:
            self.skipTest("Skipping training tests. Set `do_run_train_tests=True` to run them.")
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")

        model = AutoAdapterModel.from_config(self.config())

        name = adapter_config.__class__.__name__
        shared_name = "shared"
        # add two adapters: one will be trained and the other should be frozen
        for task in tasks:
            model.add_adapter(task, config=adapter_config)

        model.share_parameters(
            adapter_names=MultiTask(*tasks),
            name=shared_name,
        )

        dummy_tasks = [f"dummy_{task}" for task in tasks]
        dummy_shared_name = f"dummy_{shared_name}"

        for task in dummy_tasks:
            model.add_adapter(task, config=adapter_config)

        model.share_parameters(
            adapter_names=MultiTask(*dummy_tasks),
            name=dummy_shared_name,
        )

        self.add_head(model, name)
        # filter_keys.append("heads.{name}.")

        for task in tasks:
            self._assert_adapter_available(model, task)

        for task in dummy_tasks:
            self._assert_adapter_available(model, task)
        filter_keys_untrained = self._get_filter_keys(
            filter_keys,
            name=dummy_shared_name,
            task_names=dummy_tasks,
        )
        model.train_adapter(adapter_setup)
        self.assertEqual(set(tasks), model.active_adapters.flatten())

        # all weights of the adapter should be activated
        has_weights = False
        filter_keys_trained = self._get_filter_keys(filter_keys, name=shared_name, task_names=tasks)

        for k, v in self._filter_parameters(model, filter_keys_trained).items():
            has_weights = True
            self.assertTrue(v.requires_grad, k)
        self.assertTrue(has_weights)
        # all weights of the adapter not used for training should be frozen
        filter_keys_untrained = self._get_filter_keys(
            filter_keys,
            name=dummy_shared_name,
            task_names=dummy_tasks,
        )

        for k, v in self._filter_parameters(model, filter_keys_untrained).items():
            self.assertFalse(v.requires_grad, k)

        state_dict_pre = deepcopy(model.state_dict())

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

        self.run_shared_parameters_are_equals(
            model,
            shared_parameters,
            shared_name,
            tasks,
        )

        for (k1, v1), (k2, v2) in zip(state_dict_pre.items(), model.state_dict().items()):
            # move both to the same device to avoid device mismatch errors
            v1, v2 = v1.to(v2.device), v2
            if (any(key in k1 for key in filter_keys_trained) or name in k1) and not has_tied_embeddings(k1):
                adapters_with_change |= not torch.equal(v1, v2)
            else:
                base_with_change |= not torch.equal(v1, v2)

        self.assertTrue(adapters_with_change)
        self.assertFalse(base_with_change)

    def run_forward_test(
        self,
        model,
        adapter_config,
        adapter_setup,
        tasks,
        dtype=torch.float32,
        **kwargs,
    ):
        model.eval()

        for task in tasks:
            model.add_adapter(task, config=adapter_config)

        model.share_parameters(
            adapter_names=MultiTask(*tasks),
        )

        model.to(torch_device).to(dtype)

        input_data = self.get_input_samples(config=model.config, dtype=dtype, **kwargs)

        # pass 1: set adapter via property
        model.set_active_adapters(adapter_setup)
        output_1 = model(**input_data)

        # pass 2: set via context
        # unset and make sure it's unset
        model.set_active_adapters(None)
        self.assertEqual(None, model.active_adapters)
        with AdapterSetup(adapter_setup):
            output_2 = model(**input_data)

        # pass 3: base output
        model.set_active_adapters(None)
        base_output = model(**input_data)

        self.assertEqual(len(output_1), len(output_2))
        self.assertTrue(torch.equal(output_1[0], output_2[0]))
        self.assertGreaterEqual(len(output_1), len(base_output))
        self.assertFalse(torch.equal(output_1[0], base_output[0]))

        # Remove added adapters in case of multiple subtests
        model.set_active_adapters(None)
        model.unshare_parameters(
            adapter_names=MultiTask(*tasks),
        )
        for task in tasks:
            model.delete_adapter(task)

    def run_shared_parameters_are_equals(
        self,
        model,
        shared_parameters,  # name of adapter parameters / submodules
        shared_parameters_name,  # name in layer.shared_parameters which store shared parts.
        tasks,
    ):

        def test(layer, model):
            for task in tasks:
                if task in layer.adapter_modules:
                    for shared_parameter in shared_parameters:
                        adapter_params = getattr(layer.adapter_modules[task], shared_parameter)
                        shared_params = layer.shared_parameters[shared_parameters_name][shared_parameter]
                        self.assertTrue(adapter_params.equal(shared_params))

        model.apply_to_adapter_layers(lambda i, layer: test(layer, model))


@require_torch
class MultiTaskTestMixin(MultiTaskTestMethod):
    adapter_configs_to_test = [
        (
            MTLLoRAConfig(n_up_projection=3, init_weights="bert"),
            ["a", "b", "c"],
            [
                ".shared_parameters.{name}.",
                ".loras.{task}.",
                ".loras.{task}.shared_parameters.{name}.",
            ],
            ["lora_A", "lora_B"],
        ),
    ]

    def test_adapter_are_mtl_configs(self):
        model = self.get_model()
        model.eval()

        model.add_adapter("a", LoRAConfig())
        model.add_adapter("b", LoRAConfig())
        model.add_adapter("c", LoRAConfig())
        with self.assertRaises(TypeError):
            model.share_parameters(adapter_names=MultiTask("a", "b", "c"))

    @classmethod
    def generate_test_methods(cls):
        """Dynamically create test methods for each adapter config."""
        for i, (
            adapter_config,
            tasks,
            filter_keys,
            shared_parameters,
        ) in enumerate(cls.adapter_configs_to_test):

            def test_share_parameters(
                self,
                adapter_config=adapter_config,
                tasks=tasks,
                filter_keys=filter_keys,
                shared_parameters=shared_parameters,
            ):
                model = self.get_model()
                model.eval()
                with self.subTest(config=adapter_config.__class__.__name__, task_names=tasks):
                    shared_parameters_name = "shared"
                    for task in tasks:
                        model.add_adapter(task, config=adapter_config)

                    model.share_parameters(
                        adapter_names=MultiTask(*tasks),
                        name=shared_parameters_name,
                    )

                    has_weights = False
                    filter_keys = self._get_filter_keys(filter_keys, tasks, shared_parameters_name)
                    for k, v in self._filter_parameters(model, filter_keys).items():
                        has_weights = True
                        self.assertTrue(v.requires_grad, k)
                    self.assertTrue(has_weights)

                    self.run_shared_parameters_are_equals(
                        model,
                        shared_parameters,  # name of adapter parameters / submodules
                        shared_parameters_name,  # name in layer.shared_parameters which store shared parts.
                        tasks,
                    )

            def test_unshare_parameters(self, adapter_config=adapter_config, tasks=tasks):
                model = self.get_model()
                model.eval()
                with self.subTest(config=adapter_config.__class__.__name__, task_names=tasks):
                    shared_parameters_name = "shared"
                    for task in tasks:
                        model.add_adapter(task, config=adapter_config)

                    model.share_parameters(
                        adapter_names=MultiTask(*tasks),
                        name=shared_parameters_name,
                    )
                    model.unshare_parameters(
                        adapter_names=MultiTask(*tasks),
                        name=shared_parameters_name,
                    )

                    filter_keys = [f".shared_parameters.{shared_parameters_name}."]
                    parameters = self._filter_parameters(model, filter_keys)
                    self.assertTrue(parameters == {})

            def test_mtl_forward(self, adapter_config=adapter_config, tasks=tasks):
                model = self.get_model()
                model.eval()
                with self.subTest(config=adapter_config.__class__.__name__, task_names=tasks):
                    self.run_forward_test(
                        model,
                        adapter_config,
                        tasks=tasks,
                        adapter_setup=MultiTask(*tasks),
                        n_tasks=len(tasks),
                    )

            def test_mtl_train(
                self,
                adapter_config=adapter_config,
                tasks=tasks,
                filter_keys=filter_keys,
            ):
                with self.subTest(config=adapter_config.__class__.__name__, task_names=tasks):
                    self.run_train_test(
                        adapter_config,
                        filter_keys,
                        adapter_setup=MultiTask(*tasks),
                        shared_parameters=shared_parameters,
                        tasks=tasks,
                    )

            arch = adapter_config.architecture
            for method in [
                test_share_parameters,
                test_unshare_parameters,
                test_mtl_forward,
                test_mtl_train,
            ]:
                setattr(cls, f"{method.__name__}_{arch}", method)

            return cls
