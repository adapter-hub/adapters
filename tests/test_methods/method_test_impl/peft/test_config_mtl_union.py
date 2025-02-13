from accelerate.state import torch
from accelerate.utils.modeling import tempfile
from adapters.composition import MultiTask
from adapters.configuration.adapter_config import MTLLoRAConfig, MultiTaskConfigUnion
from adapters.context import ForwardContext
from adapters.utils import WEIGHTS_NAME
from huggingface_hub import os
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from tests.test_methods.method_test_impl.utils import create_twin_models
from transformers.testing_utils import require_torch, torch_device


@require_torch
class MultiTaskConfigUnionAdapterTest(AdapterMethodBaseTestMixin):

    adapter_configs_to_test = [
        (
            MultiTaskConfigUnion(
                base_config=MTLLoRAConfig(n_up_projection=3, init_weights="bert"),
                task_names=["a", "b", "c"],
            ),
            [
                "loras.shared_parameters.{name}.",
                "loras.a.",
                "loras.b.",
                "loras.c.",
            ],
        ),
    ]

    def test_add_mtl_union_adapters(self):
        model = self.get_model()
        model.eval()
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                self.run_add_test(model, adapter_config, filter_keys)

    def test_add_mtl_union_adapters_with_set_active(self):

        model = self.get_model()
        model.eval()
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                model.eval()

                name = "test_adapter_" + adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config, set_active=True)
                model.set_active_adapters == MultiTask(*adapter_config.task_names)
                model.to(torch_device)

                # adapter is correctly added to config
                self.assertTrue(name in model.adapters_config)
                self.assertEqual(adapter_config, model.adapters_config.get(name))

                # check that weights are available and active
                has_weights = False
                filter_keys = [k.format(name=name) for k in filter_keys]
                for k, v in self._filter_parameters(model, filter_keys).items():
                    has_weights = True
                    self.assertTrue(v.requires_grad, k)
                self.assertTrue(has_weights)

                # Remove added adapters in case of multiple subtests
                model.set_active_adapters(None)
                model.delete_adapter(name)

    def test_delete_mtl_union_adapters(self):
        model = self.get_model()
        model.eval()
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                self.run_delete_test(model, adapter_config, filter_keys)

    def test_load_mtl_union_adapters(self):
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(
                model_class=self.model_class.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                ForwardContext.context_args.add("task_ids")
                self.run_load_test(adapter_config, n_tasks=len(adapter_config.task_names))

    def test_mtl_union_adapter_forward(self):
        model = self.get_model()
        model.eval()
        for adapter_config, _ in self.adapter_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.base_config.__class__.__name__,
                task_names=adapter_config.task_names,
            ):
                ForwardContext.context_args.add("task_ids")
                self.run_forward_test(
                    model,
                    adapter_config,
                    n_tasks=len(adapter_config.task_names),
                    adapter_setup=MultiTask(*adapter_config.task_names),
                )

    def run_load_test(self, adapter_config, **kwargs):
        model1, model2 = create_twin_models(self.model_class, self.config)

        name = "dummy_adapter"
        model1.add_adapter(name, config=adapter_config)
        model1.set_active_adapters(name)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            model1.save_adapter(temp_dir, name)

            # Check that there are actually weights saved
            # empty string is for union shared shared params.
            for adapter_name in ["", *adapter_config.task_names]:
                weights = torch.load(
                    os.path.join(temp_dir, adapter_name, WEIGHTS_NAME),
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertTrue(len(weights) > 0)

            # also tests that set_active works
            loading_info = {}
            model2.load_adapter(temp_dir, set_active=True, loading_info=loading_info)

        # check if all weights were loaded
        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # check if adapter was correctly loaded
        for adapter_name in [name, *adapter_config.task_names]:
            self.assertTrue(adapter_name in model2.adapters_config)

        # check equal output
        input_data = self.get_input_samples(config=model1.config, **kwargs)
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0], output2[0], atol=1e-4))
