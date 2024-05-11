import math
import torch
import torch.nn as nn
from typing import Dict, List, Union

from .adapter_layer_base import AdapterLayerBase
from .modeling import Activation_Function_Class
from ..configuration.adapter_config import ReftConfig
from ..context import ForwardContext


class ReftLinear(nn.Module):
    def __init__(self, n_positions: int, in_features: int, out_features: int, bias: bool = False, orthogonal: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_positions, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        if orthogonal:
            nn.init.orthogonal_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Copied from nn.Linear
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = torch.einsum("bni,nio->bno", x, self.weight)
        if self.bias is not None:
            output += self.bias
        return output


class ReftModule(nn.Module):
    def __init__(self, in_features: int, config: ReftConfig):
        super().__init__()
        self.prefix_positions = config.prefix_positions
        self.suffix_positions = config.suffix_positions
        if config.tied_weights:
            self.projection = ReftLinear(1, in_features, config.r, orthogonal=config.orthogonality)
            self.learned_source = ReftLinear(1, in_features, config.r, bias=True)
        else:
            self.projection = ReftLinear(self.prefix_positions + self.suffix_positions, in_features, config.r, orthogonal=config.orthogonality)
            self.learned_source = ReftLinear(self.prefix_positions + self.suffix_positions, in_features, config.r, bias=True)

        if config.orthogonality:
            self.projection = nn.utils.parametrizations.orthogonal(self.projection)

        self.non_linearity = Activation_Function_Class(config.non_linearity)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        # get last non-padding token id
        context = ForwardContext.get_context()
        if hasattr(context, "seqlens"):
            last_non_padding = context.seqlens - 1
        else:
            last_non_padding = [hidden_states.size(1) - 1] * hidden_states.size(0)
        # extract prefix and suffix of seq len
        pref_ids = self.prefix_positions
        suff_ids = last_non_padding - self.suffix_positions + 1
        prefix = hidden_states[:, : pref_ids, :]
        suffix = []
        for i, suff_id in enumerate(suff_ids):
            suffix.append(hidden_states[i, suff_id : suff_id + self.suffix_positions, :])
        suffix = torch.stack(suffix, dim=0)
        adapted_states = torch.cat([prefix, suffix], dim=1)
        # apply reft
        projected_states = self.projection(adapted_states)
        source_states = self.non_linearity(self.learned_source(adapted_states))
        adapted_output = adapted_states + torch.einsum("bno,nio->bni", source_states - projected_states, self.projection.weight)
        adapted_output = self.dropout(adapted_output)

        output = []
        for i, suff_id in enumerate(suff_ids):
            output.append(torch.cat([
                adapted_output[i, : self.prefix_positions, :],
                hidden_states[i, self.prefix_positions : suff_id, :],
                adapted_output[i, self.prefix_positions :, :],
                hidden_states[i, suff_id + self.suffix_positions :, :],
            ], dim=0))

        output = torch.stack(output, dim=0)

        return output


class ReftLayer(AdapterLayerBase, nn.Module):
    adapter_modules_name = "refts"

    def __init__(self, model_config, adapters_config):
        super().__init__()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.refts = nn.ModuleDict()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        reft_config = self.adapters_config.match(
            adapter_name,
            config_type=ReftConfig,
            layer_idx=self.layer_idx,
        )
        if reft_config is not None:
            reft = ReftModule(
                self.model_config.hidden_size,
                reft_config,
            )
            reft.train(self.training)
            self.refts[adapter_name] = reft
            return True

        return False

    def forward(self, hidden_states: torch.Tensor):
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            first_adapter = adapter_setup.first()
            if first_adapter in self.refts:
                hidden_states = self.refts[first_adapter](hidden_states)

        return hidden_states


def init_reft(model):
    def hook_fn(module, args, output):
        return (module.reft_layer(output[0]),) + output[1:]

    for _, layer in model.iter_layers():
        if not hasattr(layer, "reft_layer"):
            layer.reft_layer = ReftLayer(model.config, model.adapters_config)
            layer.register_forward_hook(hook_fn)
