import os
import tempfile
import unittest

import torch

import adapters
from adapters import AdapterModelInterface, AutoAdapterModel
from adapters.utils import WEIGHTS_NAME
from parameterized import parameterized
from transformers import AutoModel, AutoModelForCausalLM, BertConfig, LlamaConfig
from transformers.testing_utils import require_torch, torch_device

from .test_adapter import ids_tensor, make_config


@require_torch
class CustomInterfaceCompatTest(unittest.TestCase):
    # This test is to check if the custom interface produces the same results as the AdapterModel implementation.

    llama_config = make_config(
        LlamaConfig,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        pad_token_id=0,
    )
    bert_config = make_config(
        BertConfig,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        pad_token_id=0,
    )
    llama_adapter_interface = AdapterModelInterface(
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
    bert_adapter_interface = AdapterModelInterface(
        adapter_types=["lora", "reft"],
        model_embeddings="embeddings",
        model_layers="encoder.layer",
        layer_self_attn="attention.self",
        layer_cross_attn=None,
        attn_k_proj="key",
        attn_q_proj="query",
        attn_v_proj="value",
        layer_intermediate_proj="intermediate.dense",
        layer_output_proj="output.dense",
    )

    def create_twin_models(self, config, adapter_interface, hf_auto_model_class):
        model1 = hf_auto_model_class.from_config(config())
        adapters.init(model1, interface=adapter_interface)
        model1.eval()
        # create a twin initialized with the same random weights
        model2 = AutoAdapterModel.from_pretrained(None, config=config(), state_dict=model1.state_dict())
        model2.eval()
        return model1, model2

    def run_load_test(self, adapter_config, config, adapter_interface, hf_auto_model_class):
        custom_model, auto_model = self.create_twin_models(config, adapter_interface, hf_auto_model_class)

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

    @parameterized.expand(
        [
            ("LoRA_Llama", adapters.LoRAConfig(), llama_config, llama_adapter_interface, AutoModelForCausalLM),
            ("LoRA_BERT", adapters.LoRAConfig(), bert_config, bert_adapter_interface, AutoModel),
            ("LoReft_Llama", adapters.LoReftConfig(), llama_config, llama_adapter_interface, AutoModelForCausalLM),
            ("LoReft_BERT", adapters.LoReftConfig(), bert_config, bert_adapter_interface, AutoModel),
        ]
    )
    def test_load_adapter(self, name, adapter_config, config, adapter_interface, hf_auto_model_class):
        self.run_load_test(adapter_config, config, adapter_interface, hf_auto_model_class)
