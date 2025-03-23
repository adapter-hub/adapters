import copy
import unittest

import pytest

import adapters
from adapters import AdapterModelInterface, ConfigUnion, DoubleSeqBnConfig, LoRAConfig, ParBnConfig
from transformers import Gemma2ForCausalLM, Gemma2ForSequenceClassification
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.testing_utils import torch_device

from .base import TextAdapterTestBase
from .generator import generate_method_tests, require_torch
from .method_test_impl.core.test_adapter_backward_compability import CompabilityTestMixin
from .method_test_impl.core.test_adapter_fusion_common import AdapterFusionModelTestMixin
from .method_test_impl.peft.test_adapter_common import BottleneckAdapterTestMixin
from .method_test_impl.utils import create_twin_models, make_config


class CustomInterfaceModelTestBase(TextAdapterTestBase):
    model_class = Gemma2ForCausalLM
    config_class = Gemma2Config
    config = make_config(
        Gemma2Config,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=16,
        pad_token_id=0,
    )
    tokenizer_name = "yujiepan/gemma-2-tiny-random"
    adapter_interface = AdapterModelInterface(
        adapter_methods=["bottleneck", "lora", "reft", "invertible"],
        model_embeddings="embed_tokens",
        model_layers="layers",
        layer_self_attn="self_attn",
        layer_cross_attn=None,
        attn_k_proj="k_proj",
        attn_q_proj="q_proj",
        attn_v_proj="v_proj",
        attn_o_proj="o_proj",
        layer_intermediate_proj="mlp.up_proj",
        layer_output_proj="mlp.down_proj",
        layer_pre_self_attn="input_layernorm",
        layer_pre_cross_attn=None,
        layer_pre_ffn="pre_feedforward_layernorm",
        layer_ln_1="post_attention_layernorm",
        layer_ln_2="post_feedforward_layernorm",
    )

    def get_model(self):
        model = Gemma2ForCausalLM(self.config())
        adapters.init(model, interface=self.adapter_interface)
        model.to(torch_device)
        return model

    def _init_model_for_train_run(self, trained_adapter_name, frozen_adapter_name, adapter_config=None):
        model = Gemma2ForSequenceClassification(self.config())
        adapters.init(model, interface=self.adapter_interface)

        model.add_adapter(
            trained_adapter_name,
            config=adapter_config or LoRAConfig(init_weights="bert"),
        )
        model.add_adapter(
            frozen_adapter_name,
            config=adapter_config or LoRAConfig(init_weights="bert"),
        )

        return model

    adapter_configs_to_test = [
        (DoubleSeqBnConfig(), ["adapters.{name}."]),
        (ParBnConfig(init_weights="bert"), ["adapters.{name}."]),
    ]

    def create_twin_models(self):
        return create_twin_models(self.model_class, self.config, self.adapter_interface)

    def test_load_mam_adapter(self):
        self.skipTest("Does not support prefix tuning.")

    def test_train_mam_adapter(self):
        self.skipTest("Does not support prefix tuning.")

    def test_merging_with_other_adapters(self):
        self.skipTest("Does not support all required methods yet.")

    def test_supports_adapter(self):
        model = self.get_model()
        model.eval()

        config = "unipelt"
        with self.assertRaises(ValueError):
            model.add_adapter("my_adapter", config=config)


method_tests = generate_method_tests(
    CustomInterfaceModelTestBase,
    not_supported=[
        "ConfigUnion",
        "ClassConversion",
        "Heads",
        "PrefixTuning",
        "PromptTuning",
        "MTLLoRA",
        "UniPELT",
        "Composition",
        "Bottleneck",
    ],
)

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class


@require_torch
@pytest.mark.bottleneck
class Bottleneck(
    CustomInterfaceModelTestBase,
    BottleneckAdapterTestMixin,
    unittest.TestCase,
):
    def test_get_adapter(self):
        model = self.get_model()
        model.eval()
        n_layers = len(list(model.iter_layers()))

        for adapter_config, n_expected in [
            (DoubleSeqBnConfig(), n_layers * 2),
            (ConfigUnion(LoRAConfig(), ParBnConfig()), n_layers * 2),
        ]:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.__class__.__name__,
            ):
                self.run_get_test(model, adapter_config, n_expected)


@require_torch
@pytest.mark.core
class Core(
    CustomInterfaceModelTestBase,
    CompabilityTestMixin,
    AdapterFusionModelTestMixin,
    unittest.TestCase,
):
    def test_wrong_interface(self):
        faulty_interface = copy.deepcopy(self.adapter_interface)
        faulty_interface.attn_k_proj = "some_non_existing_layer_name"

        model = Gemma2ForCausalLM(self.config())

        # expect ValueError
        with self.assertRaises(ValueError):
            adapters.init(model, interface=faulty_interface)
