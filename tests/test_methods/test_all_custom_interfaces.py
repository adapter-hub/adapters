import os
import tempfile
import unittest

import adapters
from adapters import CompacterPlusPlusConfig, IA3Config, LoRAConfig
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers import (
    AutoModel,
    Gemma2Config,
    Gemma3TextConfig,
    ModernBertConfig,
    PhiConfig,
    Qwen2Config,
    Qwen3Config,
)
from transformers.testing_utils import torch_device

from .base import TextAdapterTestBase


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
    "Gemma3Text": {
        "config_class": Gemma3TextConfig,
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
    "Qwen2": {
        "config_class": Qwen2Config,
        "config_params": {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "pad_token_id": 0,
        },
        "test_base": TextAdapterTestBase,
    },
    "Qwen3": {
        "config_class": Qwen3Config,
        "config_params": {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "pad_token_id": 0,
        },
        "test_base": TextAdapterTestBase,
    },
    "Phi": {
        "config_class": PhiConfig,
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


class CustomInterfaceTestBase(AdapterMethodBaseTestMixin):
    """
    Tests for the custom interfaces we support. To save time, this test suite is limited to test only basic functionality for each model.
    """

    def get_model(self):
        model = AutoModel.from_config(self.config())
        adapters.init(model)  # No need to specify interface - it's auto-detected
        model.to(torch_device)
        return model

    def test_bottleneck_forward(self):
        """Test that bottleneck forward pass works."""
        model = self.get_model()
        adapter_methods = model.base_model.adapter_interface.adapter_methods
        if "bottleneck" not in adapter_methods:
            self.skipTest("Bottleneck not supported by this model.")
        self.run_forward_test(model, "seq_bn")

    def test_invertible_forward(self):
        """Test that invertible forward pass works."""
        model = self.get_model()
        adapter_methods = model.base_model.adapter_interface.adapter_methods
        if "invertible" not in adapter_methods:
            self.skipTest("Invertible not supported by this model.")
        self.run_forward_test(model, "double_seq_bn")

    def test_prompt_tuning_forward(self):
        """Test that prompt tuning forward pass works."""
        model = self.get_model()
        adapter_methods = model.base_model.adapter_interface.adapter_methods
        if "prompt_tuning" not in adapter_methods:
            self.skipTest("Prompt tuning not supported by this model.")
        self.run_forward_test(model, "prompt_tuning")

    def test_reft_forward(self):
        """Test that reft forward pass works."""
        model = self.get_model()
        adapter_methods = model.base_model.adapter_interface.adapter_methods
        if "reft" not in adapter_methods:
            self.skipTest("Reft not supported by this model.")
        self.run_forward_test(model, "loreft")

    def test_lora_forward(self):
        """Test that lora forward pass works."""
        model = self.get_model()
        adapter_methods = model.base_model.adapter_interface.adapter_methods
        if "lora" not in adapter_methods:
            self.skipTest("LoRA not supported by this model.")
        self.run_forward_test(model, LoRAConfig(init_weights="bert", intermediate_lora=True, output_lora=True))

    def test_ia3_forward(self):
        """Test that IA3 forward pass works."""
        model = self.get_model()
        adapter_methods = model.base_model.adapter_interface.adapter_methods
        if "lora" not in adapter_methods:
            self.skipTest("IA3 not supported by this model.")
        self.run_forward_test(model, IA3Config(init_weights="bert", intermediate_lora=True, output_lora=True))

    def test_compacter_forward(self):
        """Test that compacter forward pass works."""
        model = self.get_model()
        adapter_methods = model.base_model.adapter_interface.adapter_methods
        if "bottleneck" not in adapter_methods:
            self.skipTest("Compacter not supported by this model.")
        self.run_forward_test(model, CompacterPlusPlusConfig(phm_dim=2, reduction_factor=8))

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
                lambda params=model_config["config_params"], cls=model_config["config_class"]: cls(**params)
            ),
        },
    )

    # Add test class to global namespace
    globals()[model_name] = test_class
