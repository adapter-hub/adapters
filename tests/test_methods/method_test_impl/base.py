import copy
import os
import tempfile
from typing import Callable

import torch

import adapters
from adapters import ADAPTER_MODEL_MAPPING, AdapterSetup, AdapterTrainer
from adapters.heads import CausalLMHead
from adapters.utils import WEIGHTS_NAME
from adapters.wrappers import load_model
from transformers import TrainingArguments
from transformers.testing_utils import require_torch, torch_device

from .utils import add_lm_head, create_twin_models


@require_torch
class AdapterMethodBaseTestMixin:
    """Implements base test running methods for testing adapter method implementations."""

    dtypes_to_test = [torch.float32, torch.half] if torch_device == "cuda" else [torch.float32]

    def _assert_adapter_available(self, model, adapter_name):
        """Check wether the adapter name is present in the model's adapter config and has been created."""
        self.assertTrue(adapter_name in model.adapters_config)
        self.assertGreater(len(model.get_adapter(adapter_name)), 0)

    def _assert_adapter_unavailable(self, model, adapter_name):
        """Check wether the adapter name is not present in the model's adapter config and has not been created."""
        self.assertFalse(adapter_name in model.adapters_config)
        self.assertEqual(len(model.get_adapter(adapter_name)), 0)

    def _filter_parameters(self, model, filter_keys):
        return {k: v for (k, v) in model.named_parameters() if any([filter_key in k for filter_key in filter_keys])}

    def run_add_test(self, model, adapter_config, filter_keys):
        model.eval()

        name = "test_adapter_" + adapter_config.__class__.__name__
        model.add_adapter(name, config=adapter_config)
        model.set_active_adapters(name)
        model.to(torch_device)

        # adapter is correctly added to config
        self.assertTrue(name in model.adapters_config)
        self.assertEqual(adapter_config, model.adapters_config.get(name))

        # check that weights are available and active
        has_weights = False
        filter_keys = [k.format(name=name) for k in filter_keys]
        for k, v in self._filter_parameters(model, filter_keys).items():
            has_weights = True
            self.assertTrue(v.requires_grad, k)
        self.assertTrue(has_weights)

        # Remove added adapters in case of multiple subtests
        model.set_active_adapters(None)
        model.delete_adapter(name)

    def run_leave_out_test(self, model, adapter_config, leave_out):
        model.eval()

        adapter_config = adapter_config.replace(leave_out=leave_out)
        name = "test_adapter_" + adapter_config.__class__.__name__
        model.add_adapter(name, config=adapter_config)
        model.set_active_adapters(name)

        # adapter is correctly added to config
        self._assert_adapter_available(model, name)

        adapter = model.get_adapter(name)

        self.assertNotEqual(len(adapter), 0)
        found_layers = list(adapter.keys())
        for layer in leave_out:
            self.assertNotIn(layer, found_layers)

        model.delete_adapter(name)

    def run_linear_average_test(self, model, adapter_config, filter_keys):
        model.eval()

        weights = [-0.2, 0.9, 0.3]

        # add adapters to average
        name = "test_adapter_" + adapter_config.__class__.__name__
        for i in range(len(weights)):
            model.add_adapter(name + f"_{i}", config=adapter_config)

        # collect weighted average of adapter weights
        averaged_weights = {}
        for i, w in enumerate(weights):
            this_filter_keys = [k.format(name=name + f"_{i}") for k in filter_keys]
            for k, v in self._filter_parameters(model, this_filter_keys).items():
                base_k = k.replace(name + f"_{i}", name)
                if base_k not in averaged_weights:
                    averaged_weights[base_k] = w * v
                else:
                    averaged_weights[base_k] += w * v

        # average adapters
        model.average_adapter(
            name,
            [name + f"_{i}" for i in range(len(weights))],
            weights=weights,
            combine_strategy="linear",
        )

        # adapter is correctly added to config
        self.assertTrue(name in model.adapters_config)
        self.assertEqual(adapter_config, model.adapters_config.get(name))

        # compare averaged weights to collected weights
        this_filter_keys = [k.format(name=name) for k in filter_keys]
        collected_keys = set()
        for k, v in self._filter_parameters(model, this_filter_keys).items():
            self.assertTrue(k in averaged_weights, f"Unexpected key {k} found in averaged adapter.")
            self.assertTrue(torch.allclose(v, averaged_weights[k]), k)
            collected_keys.add(k)

        # Check that the averaged weights are the same as the collected weights.
        # If something is missing here, have a look at where in the model we add weights for the averaged adapter vs where we added weights for the individual adapters.
        self.assertEqual(set(averaged_weights.keys()), collected_keys)

    def run_delete_test(self, model, adapter_config, filter_keys):
        model.eval()

        name = "test_adapter_" + adapter_config.__class__.__name__
        model.add_adapter(name, config=adapter_config)
        model.set_active_adapters(name)
        model.to(torch_device)

        # adapter is correctly added to config
        self._assert_adapter_available(model, name)

        # remove the adapter again
        model.delete_adapter(name)
        self._assert_adapter_unavailable(model, name)

        # check that weights are available and active
        has_weights = False
        filter_keys = [k.format(name=name) for k in filter_keys]
        for k, v in self._filter_parameters(model, filter_keys).items():
            has_weights = True
        self.assertFalse(has_weights)

    def run_get_test(self, model, adapter_config, num_expected_modules):
        model.eval()

        model.add_adapter("first", config=adapter_config)
        model.set_active_adapters("first")

        # adapter is correctly added to config
        name = "first"
        self._assert_adapter_available(model, name)

        adapter = model.get_adapter("first")

        self.assertNotEqual(len(adapter), 0)
        num_found_modules = sum([len(layer_modules) for layer_modules in adapter.values()])
        self.assertEqual(num_expected_modules, num_found_modules)

        model.delete_adapter("first")

    def run_forward_test(self, model, adapter_config, dtype=torch.float32, **kwargs):
        model.eval()

        name = adapter_config.__class__.__name__
        adapter_setup = kwargs.get("adapter_setup") or name
        if name not in model.adapters_config:
            model.add_adapter(name, config=adapter_config)
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
        model.delete_adapter(name)

    def create_twin_models(self):
        return create_twin_models(self.model_class, self.config)

    def run_load_test(self, adapter_config):
        model1, model2 = self.create_twin_models()

        name = "dummy_adapter"
        model1.add_adapter(name, config=adapter_config)
        model1.set_active_adapters(name)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            model1.save_adapter(temp_dir, name)

            # Check that there are actually weights saved
            weights = torch.load(
                os.path.join(temp_dir, WEIGHTS_NAME),
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
        self.assertTrue(name in model2.adapters_config)

        # check equal output
        input_data = self.get_input_samples(config=model1.config)
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0], output2[0], atol=1e-4))

    def run_full_model_load_test(self, adapter_config):
        model1 = self.get_model()
        model1.eval()

        name = "dummy"
        model1.add_adapter(name, config=adapter_config)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            model1.save_pretrained(temp_dir)

            model2, loading_info = load_model(temp_dir, self.model_class, output_loading_info=True)

        # check if all weights were loaded
        self.assertEqual(0, len(loading_info["missing_keys"]), loading_info["missing_keys"])
        self.assertEqual(
            0,
            len(loading_info["unexpected_keys"]),
            loading_info["unexpected_keys"],
        )

        # check if adapter was correctly loaded
        self.assertTrue(name in model2.adapters_config)

        # check equal output
        input_data = self.get_input_samples(config=model1.config)
        model1.to(torch_device)
        model2.to(torch_device)
        with AdapterSetup(name):
            output1 = model1(**input_data)
            output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0], output2[0], atol=1e-4))

    def trainings_run(
        self,
        model,
        lr=1.0,
        steps=8,
        batch_size=2,
        gradient_accumulation_steps=1,
    ):
        # setup dataset
        train_dataset = self.get_dataset()

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

    def run_train_test(self, adapter_config, filter_keys):
        if not self.do_run_train_tests:
            self.skipTest("Skipping training tests. Set `do_run_train_tests=True` to run them.")
        model = self._init_model_for_train_run("mrpc", "dummy", adapter_config)

        self._assert_adapter_available(model, "mrpc")
        self._assert_adapter_available(model, "dummy")

        # train the mrpc adapter -> should be activated & unfreezed
        model.train_adapter("mrpc")
        self.assertEqual(set(["mrpc"]), model.active_adapters.flatten())

        # all weights of the adapter should be activated
        has_weights = False
        filter_keys_trained = [k.format(name="mrpc") for k in filter_keys]
        for k, v in self._filter_parameters(model, filter_keys_trained).items():
            has_weights = True
            self.assertTrue(v.requires_grad, k)
        self.assertTrue(has_weights)
        # all weights of the adapter not used for training should be frozen
        filter_keys_untrained = [k.format(name="dummy") for k in filter_keys]
        for k, v in self._filter_parameters(model, filter_keys_untrained).items():
            self.assertFalse(v.requires_grad, k)

        state_dict_pre = copy.deepcopy(model.state_dict())

        self.trainings_run(model)

        # check that the adapters have changed, but the base model has not
        adapters_with_change, base_with_change = False, False
        # check whether the key corresponds to a tied embedding

        def has_tied_embeddings(k):
            tied_embeddings = hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings
            is_tied_layer = (
                hasattr(model, "heads")
                and isinstance(model.heads["mrpc"], CausalLMHead)
                and "heads.{}.{}.weight".format("mrpc", len(model.heads["mrpc"]._modules) - 1) in k
            )
            return tied_embeddings and is_tied_layer

        for (k1, v1), (k2, v2) in zip(state_dict_pre.items(), model.state_dict().items()):
            # move both to the same device to avoid device mismatch errors
            v1, v2 = v1.to(v2.device), v2
            if "mrpc" in k1 and not has_tied_embeddings(k1) or not k1.startswith(model.base_model_prefix):
                adapters_with_change |= not torch.equal(v1, v2)
            else:
                base_with_change |= not torch.equal(v1, v2)
        self.assertTrue(adapters_with_change)
        self.assertFalse(base_with_change)

    def run_merge_test(self, adapter_config):
        model = self.get_model()
        model.eval()
        model.add_adapter("test_lora", config=adapter_config)
        model.to(torch_device)

        input_data = self.get_input_samples(config=model.config)

        # forward in training mode
        model.set_active_adapters("test_lora")
        output_1 = model(**input_data)

        # forward in merged mode
        model.set_active_adapters(None)
        model.merge_adapter("test_lora")
        model.to(torch_device)
        model.eval()
        output_2 = model(**input_data)

        # check forward pass
        self.assertEqual(len(output_1), len(output_2))
        self.assertTrue(torch.allclose(output_1[0], output_2[0], atol=1e-3))

    def run_reset_test(self, adapter_config):
        model = self.get_model()
        model.eval()
        model.add_adapter("test_lora", config=adapter_config)
        model.to(torch_device)

        input_data = self.get_input_samples(config=model.config)

        # before merging
        output_1 = model(**input_data)

        # merge & reset
        model.merge_adapter("test_lora")
        model.reset_adapter()

        # after merging
        output_2 = model(**input_data)

        # check forward pass
        self.assertEqual(len(output_1), len(output_2))
        self.assertTrue(torch.allclose(output_1[0], output_2[0], atol=1e-3))

    def _run_gradient_checkpointing_test_helper(self, adapter_setup_fn: Callable[[adapters.ModelAdaptersMixin], None]):
        """
        Test that gradient checkpointing produces the same results as normal training
        Args:
            adapter_setup_fn: Function that takes a model and sets up the adapter training. Must also add a head (usually via self.add_head(...)). We have this in a separate function to allow complex setups (like training a normal adapter or training parallel setups)
        """

        if not self.do_run_train_tests:
            self.skipTest("Skipping training tests. Set `do_run_train_tests=True` to run them.")
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")

        config = self.config()
        state_dict_after_training = {}

        # Run training twice (with & without gradient checkpointing) to verify both produce identical results (i.e. the same state dict)
        for train_with_checkpointing in [True, False]:
            # Set random seed
            torch.manual_seed(42)

            # Initialize model
            model = adapters.AutoAdapterModel.from_config(config)

            # if model doesn't support gradient checkpointing, skip the test
            if not model.supports_gradient_checkpointing:
                self.skipTest("Model does not support gradient checkpointing")

            model.to(torch_device)
            adapter_setup_fn(model)

            # Enable gradient checkpointing
            if train_with_checkpointing:
                model.gradient_checkpointing_enable()

            # Train & store state dict
            self.trainings_run(model, batch_size=1, gradient_accumulation_steps=2)
            state_dict_after_training[train_with_checkpointing] = copy.deepcopy(model.state_dict())

        # Check that the state dicts are the same (we know that normal training works as expected, so we only need to check that gradient checkpointing produces the same results.)
        for (k1, v1), (k2, v2) in zip(
            state_dict_after_training[True].items(),
            state_dict_after_training[False].items(),
        ):
            v1 = v1.to(v2.device)
            self.assertTrue(
                torch.equal(v1, v2),
                msg=f"Key {k1} is not equal:\nv1: {v1}\nv2: {v2}",
            )

    def run_gradient_checkpointing_single_adapter_test(self, adapter_config):
        def adapter_setup_fn(model):
            model.add_adapter("adapter1", config=adapter_config)
            self.add_head(model, "adapter1")
            model.train_adapter("adapter1")
            model.adapter_to("adapter1", torch_device)

        self._run_gradient_checkpointing_test_helper(adapter_setup_fn)

    def run_generate_test(self, adapter_config, max_new_tokens=32):
        if self.config_class not in ADAPTER_MODEL_MAPPING or (
            "seq2seq_lm" not in ADAPTER_MODEL_MAPPING[self.config_class].head_types
            and "causal_lm" not in ADAPTER_MODEL_MAPPING[self.config_class].head_types
        ):
            self.skipTest("No seq2seq or causal language model head")
        model = self.get_model()
        model.add_adapter("generate", config=adapter_config)
        add_lm_head(self.config_class, model, "generate")
        model.set_active_adapters("generate")
        model.to(torch_device)
        generate_input = self.build_generate_input(self.input_shape).to(torch_device)
        generated = model.generate(generate_input, max_new_tokens=max_new_tokens)
        self.assertLessEqual(
            generated.shape,
            (self.input_shape[0], self.input_shape[1] + max_new_tokens),
        )

    def run_same_weights_test(self, adapter_config, filter_keys):

        # Check one model with multiple adapters with same config
        model = self.get_model()
        num_adapters = 2
        per_model_filter_keys = {}

        for i in range(num_adapters):
            model.add_adapter(f"adapter{i}", config=adapter_config)
        for i in range(num_adapters):
            name = f"adapter{i}"
            per_model_filter_keys[name] = [k.format(name=name) for k in filter_keys]

        for (k1, v1), (k2, v2) in zip(
            self._filter_parameters(model, per_model_filter_keys["adapter0"]).items(),
            self._filter_parameters(model, per_model_filter_keys["adapter1"]).items(),
        ):
            self.assertTrue(torch.equal(v1, v2), msg=f"{k1} has different weights than {k2}")

        # Check multiple models with one adapter with same config
        if hasattr(self, "adapter_interface") and self.adapter_interface:
            model1, model2 = create_twin_models(self.model_class, self.config, self.adapter_interface)
        else:
            model1, model2 = create_twin_models(self.model_class, self.config)
        model1.add_adapter("adapter", config=adapter_config)
        model2.add_adapter("adapter", config=adapter_config)
        per_model_filter_keys = {"adapter": [k.format(name="adapter") for k in filter_keys]}

        for (k1, v1), (k2, v2) in zip(
            self._filter_parameters(model1, per_model_filter_keys["adapter"]).items(),
            self._filter_parameters(model2, per_model_filter_keys["adapter"]).items(),
        ):
            self.assertTrue(torch.equal(v1, v2), msg=f"{k1} has different weights than {k2}")
