import regex as re
from tests.models.encoder_decoder.test_modeling_encoder_decoder import * # Imported to execute model tests
from transformers import AutoModelForSeq2SeqLM, BertConfig, AdapterConfig, AutoAdapterModel
from transformers.adapters.configuration import CompacterConfig

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

    def test_consistent_adapter_config(self):
        
        model = AutoModelForSeq2SeqLM.from_config(self.config())
        adapter_config = CompacterConfig(phm_dim=2, reduction_factor=8)
        model.add_adapter("test", config=adapter_config)
        # ensure that encoder and decoder actually share the shared parameters
        self.assertEqual(model.encoder.shared_parameters, model.decoder.shared_parameters)

    def test_leave_out(self):
        for leave_out in [list(range(2)), list(range(4)), list(range(6))]:
            with self.subTest(leave_out=leave_out):
                model = AutoModelForSeq2SeqLM.from_config(self.config())
                adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=4, non_linearity="gelu", leave_out=leave_out)
                model.add_adapter("test", config=adapter_config)
                num_encoder_layer = model.config.encoder.num_hidden_layers

                for name, _ in model.named_parameters():
                    if "adapter" in name:
                        layer_id = int(re.findall("layer.(\d+)", name)[0])
                        # for the decoder layers add the encoder layer number to get the layer idx
                        if name.startswith("decoder"):
                            layer_id += num_encoder_layer
                        self.assertFalse(layer_id in leave_out, name)
                