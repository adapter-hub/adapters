from tests.models.encoder_decoder.test_modeling_encoder_decoder import *  # Imported to execute model tests
from transformers import AutoModelForSeq2SeqLM, BertConfig

from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase
from .test_adapter_fusion_common import AdapterFusionModelTestMixin


class EncoderDecoderAdapterTestBase(AdapterTestBase):
    model_class = EncoderDecoderModel
    config_class = EncoderDecoderConfig
    config = staticmethod(
        lambda: EncoderDecoderConfig.from_encoder_decoder_configs(
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
            ),
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
                is_decoder=True,
                add_cross_attention=True,
            ),
        )
    )
    tokenizer_name = "bert-base-uncased"


@require_torch
class EncoderDecoderAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    EncoderDecoderAdapterTestBase,
    unittest.TestCase,
):
    def test_invertible_adapter_with_head(self):
        """This test class is copied and adapted from the identically-named test in test_adapter_heads.py."""
        model = AutoModelForSeq2SeqLM.from_config(self.config())
        model.add_adapter("test", config="pfeiffer+inv")
        model.set_active_adapters("test")

        # Set a hook before the invertible adapter to make sure it's actually called twice:
        # Once after the embedding layer and once in the prediction head.
        calls = 0

        def forward_pre_hook(module, input):
            nonlocal calls
            calls += 1

        inv_adapter = model.base_model.get_invertible_adapter()
        self.assertIsNotNone(inv_adapter)
        inv_adapter.register_forward_pre_hook(forward_pre_hook)

        in_data = self.get_input_samples((1, 128), config=model.config)
        model.to(torch_device)
        out = model(**in_data)

        self.assertEqual((1, 128, model.config.decoder.vocab_size), out[0].shape)
        self.assertEqual(2, calls)

    def test_output_adapter_gating_scores_unipelt(self):
        # TODO currently not supported
        self.skipTest("Not implemented.")

    def test_output_adapter_fusion_attentions(self):
        # TODO currently not supported
        self.skipTest("Not implemented.")
           