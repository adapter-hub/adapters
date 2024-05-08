import torch

from adapters import LoRAConfig
from transformers.testing_utils import require_torch

from .base import AdapterMethodBaseTestMixin


@require_torch
class LoRATestMixin(AdapterMethodBaseTestMixin):
    def test_add_lora(self):
        model = self.get_model()
        self.run_add_test(model, LoRAConfig(), ["loras.{name}."])

    def test_leave_out_lora(self):
        model = self.get_model()
        self.run_leave_out_test(model, LoRAConfig(), self.leave_out_layers)

    def test_merging_with_other_adapters(self):
        model = self.get_model()
        model.add_adapter("lora", config="lora")

        # Add different adapters
        model.add_adapter("bottleneck", config="seq_bn")
        model.add_adapter("prompt", config="prompt_tuning")
        model.add_adapter("prefix", config="prefix_tuning")
        model.add_adapter("ia3", config="ia3")
        model.add_adapter("unipelt", config="unipelt")
        model.add_adapter("mam", config="mam")
        model.add_adapter("compacter", config="compacter[phm_dim=2, reduction_factor=8]")

        # Merging adapters with different architectures with LoRA should raise a ValueError
        for adapter_architecture in ["bottleneck", "prompt", "prefix", "ia3", "unipelt", "mam", "compacter"]:
            with self.subTest(adapter_architecture=adapter_architecture):
                with self.assertRaises(ValueError):
                    model.average_adapter(
                        adapter_name=f"average_lora_{adapter_architecture}",
                        adapter_list=[adapter_architecture, "lora"],
                        weights=[0.5, 0.5],
                        combine_strategy="linear",
                    )

    def test_linear_average_lora(self):
        model = self.get_model()
        self.run_linear_average_test(model, LoRAConfig(), ["loras.{name}."])

    def test_linear_zhang_average_lora(self):
        # This method tests that the linear average following the Zhang et al. 2023 paper works as expected.
        # Paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html
        # This method is an adapted version of the `run_linear_average_test` method.
        model = self.get_model()
        model.eval()
        weights = [-1, 1.5, 0.5]

        # add adapters to average
        name = "test_adapter_" + LoRAConfig().__class__.__name__
        for i in range(len(weights)):
            model.add_adapter(name + f"_{i}", config=LoRAConfig())

        averaged_weights = {}
        for i, w in enumerate(weights):
            this_filter_keys = [k.format(name=name + f"_{i}") for k in ["loras.{name}."]]
            for k, v in self.filter_parameters(model, this_filter_keys).items():
                base_k = k.replace(name + f"_{i}", name)
                # Only negate the lora_B weights and use the absolute value of the weight for lora_A weights.
                weight = abs(w) if "lora_A" in k else w
                if base_k not in averaged_weights:
                    averaged_weights[base_k] = weight * v
                else:
                    averaged_weights[base_k] += weight * v

        # average adapters
        model.average_adapter(
            name, [name + f"_{i}" for i in range(len(weights))], weights=weights, combine_strategy="lora_linear_zhang"
        )

        # adapter is correctly added to config
        self.assertTrue(name in model.adapters_config)
        self.assertEqual(LoRAConfig(), model.adapters_config.get(name))

        # compare averaged weights to collected weights
        this_filter_keys = [k.format(name=name) for k in ["loras.{name}."]]
        for k, v in self.filter_parameters(model, this_filter_keys).items():
            self.assertTrue(torch.allclose(v, averaged_weights[k]), k)

    def test_linear_delta_w_svd_average_lora(self):
        model = self.get_model()
        model.eval()
        weights = [-1, 1.5, 0.5]

        # add adapters to average
        name = "test_adapter_" + LoRAConfig().__class__.__name__
        for i in range(len(weights)):
            model.add_adapter(name + f"_{i}", config=LoRAConfig())

        # average adapters
        svd_rank = 16
        model.average_adapter(
            name,
            [name + f"_{i}" for i in range(len(weights))],
            weights=weights,
            combine_strategy="lora_delta_w_svd",
            svd_rank=svd_rank,
        )

        # adapter is correctly added to config
        self.assertTrue(name in model.adapters_config)
        self.assertEqual(LoRAConfig(r=svd_rank), model.adapters_config.get(name))

        # TODO: calculate averaged weights
        # compare averaged weights to collected weights
        # this_filter_keys = [k.format(name=name) for k in ["loras.{name}."]]
        # for k, v in self.filter_parameters(model, this_filter_keys).items():
        #     self.assertTrue(torch.allclose(v, averaged_weights[k]), k)

    def test_delete_lora(self):
        model = self.get_model()
        self.run_delete_test(model, LoRAConfig(), ["loras.{name}."])

    def test_get_lora(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        self.run_get_test(model, LoRAConfig(intermediate_lora=True, output_lora=True), n_layers * 3)

    def test_forward_lora(self):
        model = self.get_model()
        for dtype in self.dtypes_to_test:
            with self.subTest(model_class=model.__class__.__name__, dtype=dtype):
                self.run_forward_test(
                    model, LoRAConfig(init_weights="bert", intermediate_lora=True, output_lora=True), dtype=dtype
                )

    def test_load_lora(self):
        self.run_load_test(LoRAConfig())

    def test_load_full_model_lora(self):
        self.run_full_model_load_test(LoRAConfig(init_weights="bert"))

    def test_train_lora(self):
        self.run_train_test(LoRAConfig(init_weights="bert"), ["loras.{name}."])

    def test_merge_lora(self):
        self.run_merge_test(LoRAConfig(init_weights="bert"))

    def test_reset_lora(self):
        self.run_reset_test(LoRAConfig(init_weights="bert"))
