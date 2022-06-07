import copy
import os
import tempfile

import torch

from transformers import AutoTokenizer, TrainingArguments
from transformers.adapters import ADAPTER_MODEL_MAPPING, AdapterSetup, AdapterTrainer, AutoAdapterModel
from transformers.adapters.utils import WEIGHTS_NAME
from transformers.testing_utils import require_torch, torch_device


def create_twin_models(model_class, config_creator=None):
    if config_creator and model_class.__name__.startswith("Auto"):
        model_config = config_creator()
        model1 = model_class.from_config(model_config)
    elif config_creator:
        model_config = config_creator()
        model1 = model_class(model_config)
    else:
        model_config = model_class.config_class()
        model1 = model_class(model_config)
    model1.eval()
    # create a twin initialized with the same random weights
    model2 = copy.deepcopy(model1)
    model2.eval()
    return model1, model2


@require_torch
class AdapterMethodBaseTestMixin:
    """Provides base test running methods for testing an adapter method implementation."""

    def filter_parameters(self, model, filter_keys):
        return {k: v for (k, v) in model.named_parameters() if any([filter_key in k for filter_key in filter_keys])}

    def run_add_test(self, model, adapter_config, filter_keys):
        model.eval()

        name = "test_adapter_" + adapter_config.__class__.__name__
        model.add_adapter(name, config=adapter_config)
        model.set_active_adapters([name])
        model.to(torch_device)

        # adapter is correctly added to config
        self.assertTrue(name in model.config.adapters)
        self.assertEqual(adapter_config, model.config.adapters.get(name))

        # check that weights are available and active
        has_weights = False
        filter_keys = [k.format(name=name) for k in filter_keys]
        for k, v in self.filter_parameters(model, filter_keys).items():
            has_weights = True
            self.assertTrue(v.requires_grad, k)
        self.assertTrue(has_weights)

    def run_delete_test(self, model, adapter_config, filter_keys):
        model.eval()

        name = "test_adapter_" + adapter_config.__class__.__name__
        model.add_adapter(name, config=adapter_config)
        model.set_active_adapters([name])
        model.to(torch_device)

        # adapter is correctly added to config
        self.assertTrue(name in model.config.adapters)
        self.assertGreater(len(model.get_adapter(name)), 0)

        # remove the adapter again
        model.delete_adapter(name)
        self.assertFalse(name in model.config.adapters)
        self.assertEqual(len(model.get_adapter(name)), 0)

        # check that weights are available and active
        has_weights = False
        filter_keys = [k.format(name=name) for k in filter_keys]
        for k, v in self.filter_parameters(model, filter_keys).items():
            has_weights = True
        self.assertFalse(has_weights)

    def run_get_test(self, model, adapter_config):
        model.eval()

        model.add_adapter("first", config=adapter_config)
        model.add_adapter("second", config=adapter_config)
        model.set_active_adapters(["first"])
        model.to(torch_device)

        # adapter is correctly added to config
        name = "first"
        self.assertTrue(name in model.config.adapters)
        self.assertEqual(adapter_config, model.config.adapters.get(name))

        first_adapter = model.get_adapter("first")
        second_adapter = model.get_adapter("second")

        self.assertNotEqual(len(first_adapter), 0)
        self.assertEqual(len(first_adapter), len(second_adapter))
        self.assertNotEqual(first_adapter, second_adapter)

        model.delete_adapter("first")
        model.delete_adapter("second")

    def run_forward_test(self, model, adapter_config):
        model.eval()

        name = adapter_config.__class__.__name__
        model.add_adapter(name, config=adapter_config)
        model.to(torch_device)

        input_data = self.get_input_samples(config=model.config)

        # pass 1: set adapter via property
        model.set_active_adapters([name])
        output_1 = model(**input_data)

        # pass 2: set via context
        # unset and make sure it's unset
        model.set_active_adapters(None)
        self.assertEqual(None, model.active_adapters)
        with AdapterSetup(name):
            output_2 = model(**input_data)

        # pass 3: base output
        model.set_active_adapters(None)
        base_output = model(**input_data)

        self.assertEqual(len(output_1), len(output_2))
        self.assertTrue(torch.equal(output_1[0], output_2[0]))
        self.assertEqual(len(output_1), len(base_output))
        self.assertFalse(torch.equal(output_1[0], base_output[0]))

    def run_load_test(self, adapter_config):
        model1, model2 = create_twin_models(self.model_class, self.config)

        name = "dummy_adapter"
        model1.add_adapter(name, config=adapter_config)
        model1.set_active_adapters([name])
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter(temp_dir, name)

            # Check that there are actually weights saved
            weights = torch.load(os.path.join(temp_dir, WEIGHTS_NAME), map_location="cpu")
            self.assertTrue(len(weights) > 0)

            # also tests that set_active works
            model2.load_adapter(temp_dir, set_active=True)

        # check if adapter was correctly loaded
        self.assertTrue(name in model2.config.adapters)

        # check equal output
        input_data = self.get_input_samples(config=model1.config)
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0], output2[0], atol=1e-4))

    def trainings_run(self, model):
        # setup dataset
        train_dataset = self.dataset()
        training_args = TrainingArguments(
            output_dir="./examples",
            do_train=True,
            learning_rate=1.0,
            max_steps=20,
            no_cuda=True,
            per_device_train_batch_size=2,
            remove_unused_columns=False,
        )

        # evaluate
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def run_train_test(self, adapter_config, filter_keys):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        model = AutoAdapterModel.from_config(self.config())

        # add two adapters: one will be trained and the other should be frozen
        model.add_adapter("mrpc", config=adapter_config)
        model.add_adapter("dummy", config=adapter_config)
        self.add_head(model, "mrpc")

        self.assertIn("mrpc", model.config.adapters.adapters)
        self.assertIn("dummy", model.config.adapters.adapters)

        # train the mrpc adapter -> should be activated & unfreezed
        model.train_adapter("mrpc")
        self.assertEqual(set(["mrpc"]), model.active_adapters.flatten())

        # all weights of the adapter should be activated
        has_weights = False
        filter_keys_trained = [k.format(name="mrpc") for k in filter_keys]
        for k, v in self.filter_parameters(model, filter_keys_trained).items():
            has_weights = True
            self.assertTrue(v.requires_grad, k)
        self.assertTrue(has_weights)
        # all weights of the adapter not used for training should be frozen
        filter_keys_untrained = [k.format(name="dummy") for k in filter_keys]
        for k, v in self.filter_parameters(model, filter_keys_untrained).items():
            self.assertFalse(v.requires_grad, k)

        state_dict_pre = copy.deepcopy(model.state_dict())

        self.trainings_run(model)

        for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items()):
            if "mrpc" in k1:
                self.assertFalse(torch.equal(v1, v2), k1)
            else:
                self.assertTrue(torch.equal(v1, v2), k1)
