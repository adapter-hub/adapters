from typing import List

import torch
import torch.nn as nn

from ..configuration.adapter_config import ReftConfig
from ..context import ForwardContext
from .adapter_layer_base import AdapterLayerBase
from .modeling import Activation_Function_Class


class ReftUnit(nn.Module):
    def __init__(
        self,
        in_dim: int,
        r_dim: int,
        orthogonal: bool = False,
        subtract_projection: bool = True,
        non_linearity: str = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.orthogonal = orthogonal
        self.learned_source = nn.Linear(in_dim, r_dim, bias=True)

        projection = nn.Linear(in_dim, r_dim, bias=False)
        if orthogonal:
            self.projection = nn.utils.parametrizations.orthogonal(projection)
        else:
            self.projection = projection

        self.subtract_projection = subtract_projection
        self.non_linearity = Activation_Function_Class(non_linearity)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        source_states = self.non_linearity(self.learned_source(x))
        if self.subtract_projection:
            projected_states = self.projection(x)
            source_states = source_states - projected_states
        adapted_output = x + torch.matmul(source_states, self.projection.weight)
        adapted_output = self.dropout(adapted_output)
        return adapted_output


class ReftModule(nn.Module):
    def __init__(self, in_features: int, config: ReftConfig):
        super().__init__()
        self.prefix_positions = config.prefix_positions
        self.suffix_positions = config.suffix_positions
        self.tied_weights = config.tied_weights
        n_units = 1 if config.tied_weights else 2
        self.units = nn.ModuleList(
            [
                ReftUnit(
                    in_features,
                    config.r,
                    config.orthogonality,
                    config.subtract_projection,
                    config.non_linearity,
                    config.dropout,
                )
                for _ in range(n_units)
            ]
        )

    def _gather_adapted_states(self, hidden_states: torch.Tensor):
        # get last non-padding token id
        context = ForwardContext.get_context()
        if hasattr(context, "seqlens"):
            last_non_padding = context.seqlens - 1
        else:
            last_non_padding = torch.tensor([hidden_states.size(1) - 1] * hidden_states.size(0))
        # extract prefix and suffix of seq len
        pref_ids = self.prefix_positions
        suff_ids = last_non_padding - self.suffix_positions + 1
        prefix = hidden_states[:, :pref_ids, :]
        suffix = []
        for i, suff_id in enumerate(suff_ids):
            suffix.append(hidden_states[i, suff_id : suff_id + self.suffix_positions, :])
        suffix = torch.stack(suffix, dim=0)
        if self.tied_weights:
            adapted_states = [torch.cat([prefix, suffix], dim=1)]
        else:
            adapted_states = [prefix, suffix]

        return adapted_states, suff_ids

    def _scatter_adapted_states(
        self, hidden_states: torch.Tensor, adapted_states: List[torch.Tensor], suff_ids: List[torch.Tensor]
    ):
        # merge prefix, suffix and adapted states
        adapted_output = torch.cat(adapted_states, dim=1)

        output = []
        for i, suff_id in enumerate(suff_ids):
            output.append(
                torch.cat(
                    [
                        adapted_output[i, : self.prefix_positions, :],
                        hidden_states[i, self.prefix_positions : suff_id, :],
                        adapted_output[i, self.prefix_positions :, :],
                        hidden_states[i, suff_id + self.suffix_positions :, :],
                    ],
                    dim=0,
                )
            )

        output = torch.stack(output, dim=0)

        return output

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        adapted_states, suff_ids = self._gather_adapted_states(hidden_states)

        # apply reft
        for i, unit in enumerate(self.units):
            adapted_states[i] = unit(adapted_states[i])

        output = self._scatter_adapted_states(hidden_states, adapted_states, suff_ids)

        return output


class ReftLayer(AdapterLayerBase, nn.Module):
    adapter_modules_name = "refts"

    def __init__(self, model_config, adapters_config):
        super().__init__()
        self.location_key = "reft"
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
        if reft_config is not None and (reft_config.layers == "all" or self.layer_idx in reft_config.layers):
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

    def pre_save_adapters(self):
        # Make sure orthogonal parametrizations are contiguous, otherwise saving with safetensors will fail
        for reft in self.refts.values():
            for unit in reft.units:
                if unit.orthogonal:
                    unit.projection.parametrizations.weight[0].base = unit.projection.parametrizations.weight[
                        0
                    ].base.contiguous()


def init_reft(model):
    def hook_fn(module, args, output):
        if isinstance(output, torch.Tensor):
            return module.reft_layer(output)
        else:
            return (module.reft_layer(output[0]),) + output[1:]

    for _, layer in model.iter_layers():
        if not hasattr(layer, "reft_layer"):
            layer.reft_layer = ReftLayer(model.config, model.adapters_config)
            layer.register_forward_hook(hook_fn)
