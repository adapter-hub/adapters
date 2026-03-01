import os
import tempfile
import unittest

import torch
import torch.nn as nn

import adapters
from adapters import CompacterPlusPlusConfig, IA3Config, LoRAConfig
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers import (
    AutoModel,
    Gemma2Config,
    Gemma3TextConfig,
    ModernBertConfig,
    PhiConfig,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2Config,
    Qwen3Config,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.testing_utils import torch_device

from .base import TextAdapterTestBase, VisionAdapterTestBase


# ---------------------------------------------------------------------------
# Vendored minimal EAT model for testing
# ---------------------------------------------------------------------------
# EATConfig is NOT in transformers (requires trust_remote_code=True on HF Hub),
# so we vendor a tiny model with the correct module hierarchy to test the
# custom interface without any external dependencies.


class _EATConfig(PretrainedConfig):
    model_type = "eat"

    def __init__(self, embed_dim=32, depth=4, num_heads=4, mlp_ratio=2, img_size=(64, 128), patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.patch_size = patch_size


class _EATAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * ((C // self.num_heads) ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _EATMlp(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _EATBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _EATAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _EATMlp(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _EATBackbone(nn.Module):
    """Mimics the inner EAT backbone (model.local_encoder, model.blocks)."""

    def __init__(self, config):
        super().__init__()
        self.local_encoder = nn.Conv2d(1, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.blocks = nn.ModuleList(
            [_EATBlock(config.embed_dim, config.num_heads, config.mlp_ratio) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, pixel_values):
        # pixel_values: (B, 1, T, F)
        x = self.local_encoder(pixel_values)  # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class _EATModel(PreTrainedModel):
    config_class = _EATConfig
    base_model_prefix = ""

    def __init__(self, config):
        super().__init__(config)
        self.model = _EATBackbone(config)

    def get_input_embeddings(self):
        raise NotImplementedError("Audio models have no token embeddings.")

    def forward(self, pixel_values, **kwargs):
        x = self.model(pixel_values)
        return BaseModelOutput(last_hidden_state=x)


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

    def test_lora_forward_all_attn_matrices(self):
        """Test LoRA with attn_matrices=["q","k","v","o"] on models with fused QKV.

        Regression test for a bug where LoRAMergedLinear.get_n_heads() counted
        "o" in attn_matrices, causing a shape mismatch in the grouped conv1d.
        The "o" projection is handled by a separate LoRALinear layer and should
        not affect the merged QKV layer's head count.
        """
        model = self.get_model()
        adapter_methods = model.base_model.adapter_interface.adapter_methods
        if "lora" not in adapter_methods:
            self.skipTest("LoRA not supported by this model.")
        self.run_forward_test(
            model,
            LoRAConfig(
                init_weights="bert",
                intermediate_lora=True,
                output_lora=True,
                attn_matrices=["q", "k", "v", "o"],
            ),
        )

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


class EATCustomInterfaceTest(CustomInterfaceTestBase, VisionAdapterTestBase, unittest.TestCase):
    """Tests for the EAT custom interface (LoRA-only audio model)."""

    config_class = _EATConfig
    input_shape = (3, 1, 64, 128)  # (batch, channels, time, freq)

    @staticmethod
    def config():
        return _EATConfig(embed_dim=32, depth=4, num_heads=4, img_size=(64, 128))

    def get_model(self):
        model = _EATModel(self.config())
        adapters.init(model)
        model.to(torch_device)
        return model
