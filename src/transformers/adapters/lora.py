# Code adapted from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py.
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .composition import AdapterCompositionBlock
from .configuration import LoRAConfig
from .layer import AdapterLayerBase


class LoRA(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
    ):
        super().__init__()
        self.r = config.r
        self.lora_alpha = config.alpha
        # Optional dropout
        if config.dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(lora_A_shape))
            self.lora_B = nn.Parameter(torch.zeros(lora_B_shape))
            self.scaling = self.lora_alpha / self.r

            if config.init_weights == "lora":
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            elif config.init_weights == "bert":
                nn.init.normal_(self.lora_A, std=0.02)
                nn.init.normal_(self.lora_B, std=0.02)
            else:
                raise ValueError("Unknown init_weights type: {}".format(config.init_weights))


class LoRALayer(AdapterLayerBase):
    def __init__(self, location_key: str, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.location_key = location_key + "_lora"
        self.config = config
        self.loras = nn.ModuleDict(dict())

        self.merged = False

    def _get_lora_shapes(self, config):
        raise NotImplementedError()

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.layer_idx = layer_idx
        lora_config = self.config.adapters.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if lora_config is not None:
            lora = LoRA(*self._get_lora_shapes(lora_config), lora_config)
            lora.train(self.training)
            self.loras[adapter_name] = lora

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.loras:
            del self.loras[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.loras:
                    for param in self.loras[name].parameters():
                        param.requires_grad = True

    def get_adapter(self, adapter_name: str) -> nn.Module:
        if adapter_name in self.loras:
            return self.loras[adapter_name]
        else:
            return None


class Linear(LoRALayer, nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        config,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs
    ):
        LoRALayer.__init__(self, location_key, config, in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def _get_lora_shapes(self, config):
        return (config.r, self.in_features), (self.out_features, config.r)

    def reset_lora(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0:
                self.weight.data -= T(lora.lora_B @ lora.lora_A) * lora.scaling
            self.merged = None

    def merge_lora(self, name: str):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                # Merge the weights and mark it
                if lora.r > 0:
                    self.weight.data += T(lora.lora_B @ lora.lora_A) * lora.scaling
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRaLayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if not self.merged:
            adapter_setup = self.get_active_setup(self.loras)
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    lora = self.loras[adapter_setup[0]]
                    if lora.r > 0:
                        result += (lora.lora_dropout(x) @ lora.lora_A.T @ lora.lora_B.T) * lora.scaling
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")

        return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(LoRALayer, nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        config,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        LoRALayer.__init__(self, location_key, config, in_features, out_features, **kwargs)

        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if any(enable_lora):
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def _get_lora_shapes(self, config):
        return (config.r * sum(self.enable_lora), self.in_features), (
            self.out_features // len(self.enable_lora) * sum(self.enable_lora),
            config.r,
        )

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def reset_lora(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    lora.lora_A.data.unsqueeze(0), lora.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * lora.scaling))
            self.merged = None

    def merge_lora(self, name: str):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                # Merge the weights and mark it
                if lora.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        lora.lora_A.data.unsqueeze(0), lora.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data += self.zero_pad(T(delta_w * lora.scaling))
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRaLayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if not self.merged:
            adapter_setup = self.get_active_setup(self.loras)
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    lora = self.loras[adapter_setup[0]]
                    if lora.r > 0:
                        after_A = F.linear(lora.lora_dropout(x), lora.lora_A)
                        after_B = F.conv1d(
                            after_A.transpose(-2, -1), lora.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                        ).transpose(-2, -1)
                        result += self.zero_pad(after_B) * lora.scaling
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")

        return F.linear(x, T(self.weight), bias=self.bias)
