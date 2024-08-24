import os
import tempfile
import unittest

import torch

import adapters
from adapters import AdapterModelInterface, AutoAdapterModel
from adapters.utils import WEIGHTS_NAME
from transformers import AutoModelForCausalLM, LlamaConfig
from transformers.testing_utils import require_torch, torch_device

from .test_adapter import ids_tensor, make_config


@require_torch
class CustomInterfaceCompatTest(unittest.TestCase):
    config = make_config(
        LlamaConfig,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        pad_token_id=0,
    )
    adapter_interface = AdapterModelInterface(
        adapter_types=["lora", "reft"],
        model_embeddings="embed_tokens",
        model_layers="layers",
        layer_self_attn="self_attn",
        layer_cross_attn=None,
        attn_k_proj="k_proj",
        attn_q_proj="q_proj",
        attn_v_proj="v_proj",
        layer_intermediate_proj="mlp.up_proj",
        layer_output_proj="mlp.down_proj",
    )

    def create_twin_models(self):
        model1 = AutoModelForCausalLM.from_config(self.config())
        adapters.init(model1, interface=self.adapter_interface)
        model1.eval()
        # create a twin initialized with the same random weights
        model2 = AutoAdapterModel.from_pretrained(None, config=self.config(), state_dict=model1.state_dict())
        model2.eval()
        return model1, model2

    def run_load_test(self, adapter_config):
        custom_model, auto_model = self.create_twin_models()

        name = "dummy_adapter"
        custom_model.add_adapter(name, config=adapter_config)
        custom_model.set_active_adapters(name)
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_model.save_adapter(temp_dir, name)

            # Check that there are actually weights saved
            weights = torch.load(os.path.join(temp_dir, WEIGHTS_NAME), map_location="cpu")
            self.assertTrue(len(weights) > 0)

            # also tests that set_active works
            loading_info = {}
            auto_model.load_adapter(temp_dir, set_active=True, loading_info=loading_info)

        # check if all weights were loaded
        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # check if adapter was correctly loaded
        self.assertTrue(name in auto_model.adapters_config)

        # check equal output
        input_data = {"input_ids": ids_tensor((2, 128), 1000)}
        custom_model.to(torch_device)
        auto_model.to(torch_device)
        output1 = custom_model(**input_data)
        output2 = auto_model(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0], output2[0], atol=1e-4))

    def test_load_lora(self):
        self.run_load_test(adapters.LoRAConfig())

    def test_load_reft(self):
        self.run_load_test(adapters.LoReftConfig())
