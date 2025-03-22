import os
import tempfile
import unittest
import adapters
from adapters import LoRAConfig
from transformers import AutoModel, Gemma2Config, ModernBertConfig
from transformers.testing_utils import torch_device

from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from .base import TextAdapterTestBase

from adapters import LoRAConfig


# To add tests for a new model, add a new entry to this dictionary. Nothing else needs to be changed.
MODEL_CONFIGS = {
    "ModernBERT": {
        "config_class": ModernBertConfig,
        "config_params": {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "pad_token_id": 0,
        },
        "test_base": TextAdapterTestBase,
    },
    "Gemma2": {
        "config_class": Gemma2Config,
        "config_params": {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 64,
            "pad_token_id": 0,
        },
        "test_base": TextAdapterTestBase,
    },
}

ADAPTER_CONFIGS = {
    "lora": LoRAConfig(init_weights="bert", intermediate_lora=True, output_lora=True),
    "reft": "loreft",
    "bottleneck": "seq_bn",
    "invertible": "double_seq_bn_inv",
    "prompt_tuning": "prompt_tuning",
}


class CustomInterfaceTestBase(AdapterMethodBaseTestMixin):
    """
    Tests for the custom interfaces we support. This test suite is limited to test the basic functionality only per model to save time.
    """

    def get_model(self):
        model = AutoModel.from_config(self.config())
        adapters.init(model)  # No need to specify interface, it's auto-detected
        model.to(torch_device)
        return model

    def test_forward_pass(self):
        """Test that forward pass works with all supported adapter methods."""
        model = self.get_model()
        adapter_methods = model.adapter_interface.adapter_methods

        for method in adapter_methods:
            with self.subTest(adapter_method=method):
                self.run_forward_test(model, ADAPTER_CONFIGS[method])

    def test_adapter_loading(self):
        """Test saving and loading adapters."""
        model = self.get_model()
        model.eval()

        # Add and activate adapter
        name = "test_adapter"
        model.add_adapter(name, config=LoRAConfig())
        model.set_active_adapters(name)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save adapter
            model.save_adapter(temp_dir, name)
            # Check that weights file exists
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "pytorch_adapter.bin")))

            # Load adapter into new model
            new_model = self.get_model()
            new_model.load_adapter(temp_dir)
            self.assertTrue(name in new_model.adapters_config)


# Create test classes for each model
for model_name, model_config in MODEL_CONFIGS.items():
    # Create model-specific test class
    test_class = type(
        model_name,  # This will create classes named "ModernBERT" and "Gemma2"
        (CustomInterfaceTestBase, model_config["test_base"], unittest.TestCase),  # Include the specified test base
        {
            "config_class": model_config["config_class"],
            "config": staticmethod(
                lambda params=model_config["config_params"]: model_config["config_class"](**params)
            ),
        },
    )

    # Add test class to global namespace
    globals()[model_name] = test_class
