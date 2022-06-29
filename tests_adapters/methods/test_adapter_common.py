import copy
import tempfile

import torch

from transformers import (
    ADAPTER_CONFIG_MAP,
    ADAPTER_MODEL_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoAdapterModel,
    AutoTokenizer,
    HoulsbyConfig,
    HoulsbyInvConfig,
    InvertibleAdaptersMixin,
    MAMConfig,
    PfeifferConfig,
    PfeifferInvConfig,
)
from transformers.adapters import BatchSplit, Fuse
from transformers.testing_utils import require_torch, torch_device

from .base import AdapterMethodBaseTestMixin, create_twin_models


@require_torch
class BottleneckAdapterTestMixin(AdapterMethodBaseTestMixin):

    adapter_configs_to_test = [
        (PfeifferConfig(), ["adapters.{name}."]),
        (MAMConfig(), ["adapters.{name}.", "prefix_tunings.{name}."]),
    ]

    def test_add_adapter(self):
        model = self.get_model()
        model.eval()

        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_add_test(model, adapter_config, filter_keys)

    def test_delete_adapter(self):
        model = self.get_model()
        model.eval()

        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_delete_test(model, adapter_config, filter_keys)

    def test_add_adapter_with_invertible(self):
        model = self.get_model()
        model.eval()
        if not isinstance(model, InvertibleAdaptersMixin):
            self.skipTest("Model does not support invertible adapters.")

        for adapter_config in [PfeifferInvConfig(), HoulsbyInvConfig()]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config)
                model.set_active_adapters([name])

                # adapter is correctly added to config
                self.assertTrue(name in model.config.adapters)
                self.assertEqual(adapter_config, model.config.adapters.get(name))

                # invertible adapter is correctly added and returned
                self.assertTrue(name in model.invertible_adapters)
                self.assertEqual(model.invertible_adapters[name], model.get_invertible_adapter())

                # all invertible adapter weights should be activated for training
                for param in model.invertible_adapters[name].parameters():
                    self.assertTrue(param.requires_grad)

                # check forward pass
                input_data = self.get_input_samples(config=model.config)
                model.to(torch_device)
                adapter_output = model(**input_data)
                # make sure the output is different without invertible adapter
                del model.invertible_adapters[name]
                adapter_output_no_inv = model(**input_data)
                self.assertEqual(len(adapter_output), len(adapter_output_no_inv))
                self.assertFalse(torch.equal(adapter_output[0], adapter_output_no_inv[0]))

    def test_get_adapter(self):
        model = self.get_model()
        model.eval()

        for adapter_config, _ in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_get_test(model, adapter_config)

    def test_add_adapter_multiple_reduction_factors(self):
        model = self.get_model()
        model.eval()
        reduction_factor = {"1": 1, "default": 2}
        for adapter_config in [
            PfeifferConfig(reduction_factor=reduction_factor),
            HoulsbyConfig(reduction_factor=reduction_factor),
        ]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config)
                model.set_active_adapters([name])

                # adapter is correctly added to config
                self.assertTrue(name in model.config.adapters)
                self.assertEqual(adapter_config, model.config.adapters.get(name))

                adapter = model.get_adapter(name)

                self.assertEqual(
                    adapter[0]["output_adapter"].adapter_down[0].in_features
                    / adapter[0]["output_adapter"].adapter_down[0].out_features,
                    reduction_factor["default"],
                )
                self.assertEqual(
                    adapter[1]["output_adapter"].adapter_down[0].in_features
                    / adapter[1]["output_adapter"].adapter_down[0].out_features,
                    reduction_factor["1"],
                )

    def test_reduction_factor_no_default(self):
        model = self.get_model()
        model.eval()
        reduction_factor = {"2": 8, "4": 32}
        for adapter_config in [
            PfeifferConfig(reduction_factor=reduction_factor),
            HoulsbyConfig(reduction_factor=reduction_factor),
        ]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                with self.assertRaises(KeyError):
                    model.add_adapter(name, config=adapter_config)

    def test_adapter_forward(self):
        model = self.get_model()
        model.eval()

        for adapter_config, _ in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_forward_test(model, adapter_config)

    def test_load_adapter(self):
        self.run_load_test(PfeifferConfig())

    def test_load_mam_adapter(self):
        self.run_load_test(MAMConfig())

    def test_load_full_model_adapter(self):
        self.run_full_model_load_test(PfeifferConfig())

    def test_model_config_serialization(self):
        """PretrainedConfigurations should not raise an Exception when serializing the config dict

        See, e.g., PretrainedConfig.to_json_string()
        """
        for k, v in ADAPTER_CONFIG_MAP.items():
            model = self.get_model()
            # HACK: reduce the reduction factor such that
            # the small test model can have a phm_dim of 4
            if hasattr(v, "phm_layer") and v.phm_layer:
                v = v.__class__(reduction_factor=4)
            model.add_adapter("test", config=v)
            # should not raise an exception
            model.config.to_json_string()

    def test_model_adapter_summary(self):
        # count model parameters before
        model = self.get_model()
        model_no_params = sum(p.numel() for p in model.parameters())
        for k, v in ADAPTER_CONFIG_MAP.items():
            # HACK: reduce the reduction factor such that
            # the small test model can have a phm_dim of 4
            if hasattr(v, "phm_layer") and v.phm_layer:
                v = v.__class__(reduction_factor=4)
            model.add_adapter(k, config=v)
        summary = model.adapter_summary(as_dict=True)
        self.assertEqual(len(ADAPTER_CONFIG_MAP) + 1, len(summary))
        for name in ADAPTER_CONFIG_MAP.keys():
            self.assertTrue(any([row["name"] == name for row in summary]))
        self.assertEqual(model_no_params, summary[-1]["#param"])

    def test_loading_adapter_weights_with_prefix(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")

        model_base, model_with_head_base = create_twin_models(self.model_class, self.config)

        model_with_head = AutoAdapterModel.from_config(model_with_head_base.config)
        setattr(model_with_head, model_with_head.base_model_prefix, model_with_head_base)

        model_with_head.add_adapter("dummy")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_with_head.save_adapter(temp_dir, "dummy")

            loading_info = {}
            model_base.load_adapter(temp_dir, loading_info=loading_info)

        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # check equal output
        input_data = self.get_input_samples(config=model_with_head.config)
        model_with_head.to(torch_device)
        model_base.to(torch_device)
        output1 = model_with_head(**input_data)
        output2 = model_base(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_loading_adapter_weights_without_prefix(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")

        model_base, model_with_head_base = create_twin_models(self.model_class, self.config)

        model_with_head = AutoAdapterModel.from_config(model_with_head_base.config)
        setattr(model_with_head, model_with_head.base_model_prefix, model_with_head_base)

        model_base.add_adapter("dummy")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_base.save_adapter(temp_dir, "dummy")

            loading_info = {}
            model_with_head.load_adapter(temp_dir, loading_info=loading_info)

        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # check equal output
        input_data = self.get_input_samples(config=model_with_head.config)
        model_with_head.to(torch_device)
        model_base.to(torch_device)
        output1 = model_with_head(**input_data)
        output2 = model_base(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_forward_with_past(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        if self.config_class not in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
            self.skipTest("No causal lm class.")

        static_model = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[self.config_class](self.config())
        flex_model = AutoAdapterModel.from_pretrained(
            None, config=self.config(), state_dict=static_model.state_dict()
        )
        static_model.add_adapter("dummy")
        static_model.set_active_adapters("dummy")
        static_model.eval()
        flex_model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            static_model.save_adapter(temp_dir, "dummy")

            loading_info = {}
            flex_model.load_adapter(temp_dir, loading_info=loading_info)
            flex_model.set_active_adapters("dummy")

        input_data = self.get_input_samples(config=static_model.config)
        static_model.eval()
        flex_model.eval()
        static_model.to(torch_device)
        flex_model.to(torch_device)
        output = static_model(**input_data)

        input_data["past_key_values"] = output["past_key_values"]
        output_base = static_model(**input_data)
        output_with_head = flex_model(**input_data)
        self.assertTrue(torch.allclose(output_base["logits"], output_with_head["logits"]))

    def test_train_single_adapter(self):
        self.run_train_test(PfeifferConfig(), ["adapters.{name}."])

    def test_train_mam_adapter(self):
        self.run_train_test(MAMConfig(), ["adapters.{name}."])

    def test_train_adapter_fusion(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        model = AutoAdapterModel.from_config(self.config())
        self.add_head(model, "head")

        # add the adapters to be fused
        model.add_adapter("a")
        model.add_adapter("b")
        model.add_adapter("c")

        self.assertIn("a", model.config.adapters.adapters)
        self.assertIn("b", model.config.adapters.adapters)
        self.assertIn("c", model.config.adapters.adapters)

        # setup fusion
        adapter_setup = Fuse("a", "b", "c")
        model.add_adapter_fusion(adapter_setup)
        model.train_adapter_fusion(adapter_setup)
        model.set_active_adapters(adapter_setup)
        self.assertEqual(adapter_setup, model.active_adapters)

        # all weights of the adapters should be frozen (test for one)
        for k, v in self.filter_parameters(model, ["adapters.a."]).items():
            self.assertFalse(v.requires_grad, k)
        # all weights of the fusion layer should be activated
        for k, v in self.filter_parameters(model, ["adapter_fusion_layer"]).items():
            self.assertTrue(v.requires_grad, k)
        # weights of the model should be frozen (check on some examples)
        for k, v in self.filter_parameters(model, ["encoder.layer.0.attention"]).items():
            self.assertFalse(v.requires_grad, k)

        state_dict_pre = copy.deepcopy(model.state_dict())

        # Since our config has a value matrix, make sure it is regularized.
        # We do this by patching the fusion regularization function.
        regularization_called = False
        orig_fusion_regularization_loss = model.base_model.get_fusion_regularization_loss

        def patched_fusion_reg_loss():
            nonlocal regularization_called
            regularization_called = True
            return orig_fusion_regularization_loss()

        model.base_model.get_fusion_regularization_loss = patched_fusion_reg_loss

        self.trainings_run(model)

        for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items()):
            if (
                "adapter_fusion_layer" in k1
                or "classifier" in k1
                or "classification_head" in k1
                or "score" in k1
                or "heads" in k1
            ):
                self.assertFalse(torch.equal(v1, v2), k1)
            else:
                self.assertTrue(torch.equal(v1, v2), k1)
        self.assertTrue(regularization_called)

    def test_batch_split_training(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        model = AutoAdapterModel.from_config(self.config())

        model.add_adapter("mrpc1")
        model.add_adapter("mrpc2")
        self.add_head(model, "mrpc1")
        self.add_head(model, "mrpc2")
        adapter_setup = BatchSplit("mrpc1", "mrpc2", batch_sizes=[1, 1])
        model.active_adapters = adapter_setup
        model.train_adapter(adapter_setup)

        # all weights of the adapter should be activated
        for k, v in self.filter_parameters(model, ["adapters.mrpc1."]).items():
            self.assertTrue(v.requires_grad, k)
        # all weights of the adapter not used for training should be frozen
        for k, v in self.filter_parameters(model, ["adapters.mrpc2."]).items():
            self.assertTrue(v.requires_grad, k)
        # weights of the model should be frozen (check on some examples)
        for k, v in self.filter_parameters(model, ["encoder.layer.0.attention"]).items():
            self.assertFalse(v.requires_grad, k)

        state_dict_pre = copy.deepcopy(model.state_dict())

        self.trainings_run(model)

        self.assertFalse(
            all(
                torch.equal(v1, v2)
                for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items())
                if "mrpc" in k1
            )
        )
        self.assertTrue(
            all(
                torch.equal(v1, v2)
                for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items())
                if "mrpc" not in k1
            )
        )
