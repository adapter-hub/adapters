from adapters.composition import MultiTask
from adapters.configuration.adapter_config import (
    MTLLoRAConfig,
    MultiTaskConfigUnion,
)
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch, torch_device


@require_torch
class MultiTaskConfigUnionAdapterTest(AdapterMethodBaseTestMixin):

    adapter_configs_to_test = [
        (
            MultiTaskConfigUnion(
                base_config=MTLLoRAConfig(), task_names=["a", "b", "c"]
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
                model.set_active_adapters == MultiTask(
                    *adapter_config.task_names
                )
                model.to(torch_device)

                # adapter is correctly added to config
                self.assertTrue(name in model.adapters_config)
                self.assertEqual(
                    adapter_config, model.adapters_config.get(name)
                )

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
                self.run_load_test(adapter_config)

    # def test_mtl_union_adapter_forward(self):
    #     model = self.get_model()
    #     model.eval()
    #     for adapter_config, _ in self.adapter_configs_to_test:
    #         with self.subTest(
    #             model_class=model.__class__.__name__,
    #             config=adapter_config.base_config.__class__.__name__,
    #             task_names=adapter_config.task_names,
    #         ):
    #             self.run_forward_test(
    #                 model,
    #                 adapter_config,
    #                 adapter_setup=MultiTask(*adapter_config.task_names),
    #             )
