import unittest

import adapters
from adapters import AdapterModelInterface
from transformers import AutoModelForCausalLM
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.testing_utils import require_torch, torch_device

from .methods import IA3TestMixin, LoRATestMixin, ReftTestMixin, create_twin_models
from .test_adapter import AdapterTestBase, make_config


class CustomInterfaceModelTestBase(AdapterTestBase):
    model_class = AutoModelForCausalLM
    config_class = Gemma2Config
    config = make_config(
        Gemma2Config,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=16,
        head_dim=2,
        num_key_value_heads=2,
        pad_token_id=0,
    )
    tokenizer_name = "yujiepan/gemma-2-tiny-random"
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

    def get_model(self):
        model = AutoModelForCausalLM.from_config(self.config())
        adapters.init(model, interface=self.adapter_interface)
        model.to(torch_device)
        return model


@require_torch
class CustomInterfaceModelTest(
    # BottleneckAdapterTestMixin,
    # CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    # PrefixTuningTestMixin,
    # PromptTuningTestMixin,
    ReftTestMixin,
    # UniPELTTestMixin,
    # EmbeddingTestMixin,
    # AdapterFusionModelTestMixin,
    # CompabilityTestMixin,
    # ParallelAdapterInferenceTestMixin,
    # ParallelTrainingMixin,
    CustomInterfaceModelTestBase,
    unittest.TestCase,
):
    def create_twin_models(self):
        return create_twin_models(self.model_class, self.config, self.adapter_interface)
