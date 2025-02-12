from accelerate.state import torch
from accelerate.utils.modeling import tempfile
from adapters.composition import MultiTaskLearning
from adapters.configuration.adapter_config import MTLConfigUnion, MTLLoRAConfig
from adapters.context import ForwardContext
from adapters.utils import WEIGHTS_NAME
from huggingface_hub import os
from tests.methods.base import AdapterMethodBaseTestMixin, create_twin_models
from transformers.testing_utils import require_torch, torch_device


@require_torch
class ConfigMultiTaskAdapterTest(AdapterMethodBaseTestMixin):
    mtl_configs_to_test = [
        (
            MTLConfigUnion(
                MTLLoRAConfig(), task_names=["lora1", "lora2", "lora3"]
            ),
            [
                "loras.shared_parameters.{name}.",
                "loras.lora1.",
                "loras.lora2.",
                "loras.lora3.",
            ],
        )
    ]

    def test_add_mtl_adapter(self):
        model = self.get_model()
        model.eval()

        for (
            adapter_config,
            adapter_filter_keys,
        ) in self.mtl_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.__class__.__name__,
            ):
                self.run_add_test(
                    model,
                    adapter_config,
                    adapter_filter_keys,
                )

    def test_delete_mtl_adapter(self):
        model = self.get_model()
        model.eval()

        for (
            adapter_config,
            adapter_filter_keys,
        ) in self.mtl_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.__class__.__name__,
            ):
                self.run_delete_test(
                    model,
                    adapter_config,
                    adapter_filter_keys,
                    composition=MultiTaskLearning(*adapter_config.task_names),
                )

    def test_load_mtl_adapter(self):
        model = self.get_model()
        model.eval()

        for (
            adapter_config,
            adapter_filter_keys,
        ) in self.mtl_configs_to_test:
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.__class__.__name__,
            ):
                model1, model2 = create_twin_models(
                    self.model_class, self.config
                )

                name = "dummy_adapter"
                model1.add_adapter(name, config=adapter_config)
                model1.set_active_adapters(
                    MultiTaskLearning(*adapter_config.task_names)
                )
                with tempfile.TemporaryDirectory() as temp_dir:
                    model1.save_mtl_adapters(temp_dir, name)
                    # Check that there are actually weights saved
                    for task in model1.active_adapters.children:
                        weights = torch.load(
                            os.path.join(temp_dir, task, WEIGHTS_NAME),
                            map_location="cpu",
                        )
                        self.assertTrue(len(weights) > 0)

                    # also tests that set_active works
                    loading_info = {}
                    model2.load_mtl_adapters(
                        temp_dir, set_active=True, loading_info=loading_info
                    )

                # check if all weights were loaded
                self.assertEqual(0, len(loading_info["missing_keys"]))
                self.assertEqual(0, len(loading_info["unexpected_keys"]))

                # check if adapter was correctly loaded
                for task in model1.active_adapters.children:
                    self.assertTrue(task in model2.adapters_config)

                # check equal output
                input_data = self.get_input_samples(config=model1.config)
                # TODO: add forward context task_ids
                model1.to(torch_device)
                model2.to(torch_device)
                output1 = model1(**input_data)
                output2 = model2(**input_data)
                self.assertEqual(len(output1), len(output2))
                self.assertTrue(
                    torch.allclose(output1[0], output2[0], atol=1e-4)
                )
