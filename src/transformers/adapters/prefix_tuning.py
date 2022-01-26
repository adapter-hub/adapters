from typing import List, Union

import torch
from torch import nn

from .composition import AdapterCompositionBlock, Stack
from .configuration import PrefixTuningConfig
from .context import AdapterSetup
from .layer import AdapterLayerBase
from .modeling import Activation_Function_Class


class PrefixTuning(nn.Module):
    def __init__(
        self,
        n_heads: int,
        input_size: int,
        config: PrefixTuningConfig,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = self.input_size // self.n_heads
        self.config = config

        self.input_tokens = torch.arange(self.config.prefix_length).long()
        self.wte = nn.Embedding(self.config.prefix_length, self.input_size)
        self.control_trans = nn.Sequential(
            nn.Linear(self.input_size, self.config.bottleneck_size),
            Activation_Function_Class(self.config.non_linearity.lower()),
            nn.Linear(self.config.bottleneck_size, 2 * self.input_size),
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, batch_size):
        device = next(self.parameters()).device
        input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x 2*input_size
        key_values = key_values.view(
            batch_size, self.config.prefix_length, 2, self.n_heads, self.n_embd_per_head
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # 2 x batch_size x n_heads x prefix_length x n_embd_per_head
        key_values = key_values.permute(2, 0, 3, 1, 4)

        return key_values


class FlatPrefixTuning(nn.Module):
    def __init__(
        self,
        n_heads: int,
        input_size: int,
        config: PrefixTuningConfig,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = self.input_size // self.n_heads
        self.config = config

        self.wte = nn.Embedding(self.config.prefix_length, self.input_size)
        self.control_trans = nn.Parameter(torch.randn(self.config.prefix_length * 2 * self.input_size))

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, batch_size):
        key_values = self.control_trans.view(
            batch_size, self.config.prefix_length, 2, self.n_heads, self.n_embd_per_head
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # 2 * (batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


class PrefixTuningLayer(AdapterLayerBase):
    """
    Models one layer in a model containing prefix tuning modules.

    Args:
        location_key (str): The id describing the location of this layer in the model.
                            Currently, can be "encoder_prefix", "cross_prefix" or None.
        config (:class:`~transformers.PretrainedConfig`): The model config.
    """

    def __init__(self, location_key: str, config):
        super().__init__()
        self.config = config
        self.location_key = location_key
        self.prefix_tunings = nn.ModuleDict()

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.layer_idx = layer_idx
        # only match location keys for which we have config keys
        if self.location_key.startswith("cross") or self.location_key.startswith("encoder"):
            used_location_key = self.location_key
        else:
            used_location_key = None
        prefix_tuning_config = self.config.adapters.match(
            adapter_name,
            config_type=PrefixTuningConfig,
            layer_idx=self.layer_idx,
            location_key=used_location_key,
        )
        if prefix_tuning_config is not None:
            if prefix_tuning_config["flat"]:
                prefix_tuning = FlatPrefixTuning(
                    n_heads=self.config.num_attention_heads,
                    input_size=self.config.hidden_size,
                    config=prefix_tuning_config,
                )
            else:
                prefix_tuning = PrefixTuning(
                    n_heads=self.config.num_attention_heads,
                    input_size=self.config.hidden_size,
                    config=prefix_tuning_config,
                )
            prefix_tuning.train(self.training)  # make sure training mode is consistent
            self.prefix_tunings[adapter_name] = prefix_tuning

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.prefix_tunings:
            del self.prefix_tunings[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to prefix tuning

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to prefix tuning

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for prefix_tuning_name in adapter_setup.flatten():
                if prefix_tuning_name in self.prefix_tunings:
                    for param in self.prefix_tunings[prefix_tuning_name].parameters():
                        param.requires_grad = True

    def get_adapter(self, adapter_name):
        if adapter_name in self.prefix_tunings:
            return self.prefix_tunings[adapter_name]
        else:
            return None

    def forward(self, key_states, value_states, attention_mask=None):
        if hasattr(self.config, "adapters"):
            # First check current context before falling back to defined setup
            context = AdapterSetup.get_context()
            if context is not None:
                adapter_setup = context.adapter_setup
            else:
                adapter_setup = self.config.adapters.active_setup
        else:
            adapter_setup = None
        skip_adapters = adapter_setup is None or (
            self.config.adapters.skip_layers is not None and self.layer_idx in self.config.adapters.skip_layers
        )
        if not skip_adapters and (len(set(self.prefix_tunings.keys()) & adapter_setup.flatten()) > 0):
            if isinstance(adapter_setup, Stack) and len(adapter_setup) == 1:
                # we already made sure we only have 1 item
                prefix_tuning_name = adapter_setup.first()
                if prefix_tuning_name in self.prefix_tunings:
                    prefix_tuning = self.prefix_tunings[prefix_tuning_name]
                    batch_size = key_states.size(0)
                    prefix_keys, prefix_values = prefix_tuning(batch_size)

                    key_states = torch.cat([prefix_keys, key_states], dim=2)
                    value_states = torch.cat([prefix_values, value_states], dim=2)
                    if attention_mask is not None:
                        if attention_mask.dim() == 2:
                            prefix_mask = torch.ones(batch_size, prefix_keys.size(2)).to(attention_mask.device)
                        else:
                            prefix_mask = torch.ones(batch_size, 1, attention_mask.size(2), prefix_keys.size(2)).to(
                                attention_mask.device
                            )
                        attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            else:
                raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with prefix tuning.")

        return key_states, value_states, attention_mask
