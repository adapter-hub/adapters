# Code adapted from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py.
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
from copy import deepcopy
from typing import Dict, List, NamedTuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.configuration.adapter_config import MTLLoRAConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.pytorch_utils import Conv1D

from ..composition import Average, BatchSplit, MultiTask, Parallel, Stack
from ..configuration import LoRAConfig, ModelAdaptersConfig
from ..context import ForwardContext
from ..utils import multigetattr, multisetattr
from .adapter_layer_base import AdapterLayerBase, ComposableAdapterLayerBase
from .utils import dequantize_bnb_weight, fix_seed


try:
    from bitsandbytes.nn import Int8Params, Linear4bit, Linear8bitLt, Params4bit

    bitsandbytes_available = True
except ImportError:
    bitsandbytes_available = False

logger = logging.getLogger(__name__)


class TaskSpecificSingularValue(nn.Module):
    def __init__(self, rank: int, dtype: Optional[str] = None):
        super().__init__()
        self.rank = rank
        self.dtype = dtype
        self.values = nn.Parameter(torch.ones(rank, dtype=dtype))

    def forward(self):
        return torch.diag(self.values)


class TaskSpecificLinear(nn.Module):
    def __init__(self, rank: int, dtype: Optional[str] = None):
        self.rank = (rank,)
        self.dtype = dtype
        self.values = nn.Parameter(torch.ones(rank, rank, dtype=dtype))

    def forward(self):
        return self.values


TASK_SPECIFIC_MATRIX_CLS = {
    "singular_values": TaskSpecificSingularValue,
    "linear": TaskSpecificLinear,
}


class LoRA(nn.Module):
    sharable_parameters = [
        "lora_A",
        "lora_B",
    ]

    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
        name: str = None,
        **kwargs,
    ):
        super().__init__()
        assert config.composition_mode == "add", "LoRA module only supports composition_mode='add'."
        self.r = config.r
        self.lora_alpha = config.alpha
        self.composition_mode = config.composition_mode
        self.attn_matrices = config.attn_matrices
        self.use_gating = config.use_gating
        self.name = name
        self.use_dora = config.use_dora

        # Optional dropout
        if config.dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = lambda x: x

        self.dtype = getattr(torch, config.dtype) if config.dtype else None
        # Actual trainable parameters
        self.lora_A = nn.Parameter(torch.zeros(lora_A_shape, dtype=self.dtype))

        self.lora_B = nn.Parameter(torch.zeros(lora_B_shape, dtype=self.dtype))
        self.scaling = self.lora_alpha / self.r

        # Set seed for reproducibility if specified in config
        fix_seed(config.init_weights_seed)

        # For compatibility with (IA)^3, allow all init_weights types here.
        # Usually should be "lora".
        if config.init_weights == "lora":
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        elif config.init_weights == "bert":
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.normal_(self.lora_B, std=0.02)
        elif config.init_weights == "ia3":
            nn.init.ones_(self.lora_A)
            nn.init.ones_(self.lora_B)
        elif config.init_weights == "vera":
            nn.init.kaiming_uniform_(self.lora_A)
            nn.init.kaiming_uniform_(self.lora_B)
        else:
            raise ValueError("Unknown init_weights type: {}".format(config.init_weights))

        if self.use_dora:
            weights = kwargs["w_o"].clone().detach()
            weights = torch.linalg.norm(weights, dim=1).unsqueeze(1).to(dtype=self.dtype)
            self.m = nn.Linear(1, lora_B_shape[0], bias=False, dtype=self.dtype)
            with torch.no_grad():
                self.m.weight.copy_(weights)
        if self.use_gating:
            self.gate = nn.Linear(lora_A_shape[-1], gating_heads)
            nn.init.normal_(self.gate.weight, std=0.02)

    @property
    def delta_w(self) -> torch.Tensor:
        return self.lora_B @ self.lora_A

    def com(self, weights: torch.Tensor, added: torch.Tensor, scaling=None) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if scaling is None:
            scaling = self.scaling
        return weights + added * scaling

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        return weights - added * self.scaling

    def get_parameters(self, parameters_names=None):
        parameters_names = parameters_names or self.sharable_parameters
        return nn.ParameterDict(
            {
                param_name: deepcopy(getattr(self, param_name))
                for param_name in parameters_names
                if hasattr(self, param_name) and param_name in self.sharable_parameters
            }
        )

    def set_parameters(self, parameters):
        for name, param in parameters.items():
            if name in self.sharable_parameters:
                setattr(self, name, param)

    def forward(self, hidden_states: Optional[torch.Tensor], layer_input: torch.Tensor):
        if hidden_states is None:
            hidden_states = layer_input
        hidden_states = self.lora_dropout(hidden_states) @ torch.t(self.lora_A) @ torch.t(self.lora_B)
        if self.use_gating:
            gate = torch.sigmoid(self.gate(layer_input))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            hidden_states = hidden_states * gate
        else:
            gate = None
            hidden_states = hidden_states * self.scaling

        return hidden_states, gate


class MTLLoRA(LoRA):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: MTLLoRAConfig,
        gating_heads: int = 1,
        **kwargs,
    ):
        super().__init__(
            lora_A_shape,
            lora_B_shape,
            config,
            gating_heads,
        )
        self.lora_B = nn.Parameter(torch.zeros(config.n_up_projection, *lora_B_shape, dtype=self.dtype))
        if config.init_weights == "lora":
            nn.init.zeros_(self.lora_B)
        elif config.init_weights == "bert":
            nn.init.normal_(self.lora_B, std=0.02)

        self.task_specific = TASK_SPECIFIC_MATRIX_CLS[config.task_specific_matrix_type](config.r)

        self.weights = nn.Parameter(torch.ones(config.n_up_projection))
        self.weights_sharpness = config.weights_sharpness

    def apply_weights(self, hidden_states: torch.Tensor):
        w = (self.weights / self.weights_sharpness).exp()
        # hidden_states : n_up_projection x bsz x seq_len x dim
        return torch.sum(((w[:, None, None, None] * hidden_states) / w.sum()), 0)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        layer_input: torch.Tensor,
    ):
        if hidden_states is None:
            hidden_states = layer_input

        hidden_states = (
            self.lora_dropout(hidden_states)
            @ torch.t(self.lora_A)
            @ self.task_specific()
            @ torch.transpose(self.lora_B.unsqueeze(1), -1, -2)
        )
        # apply weights
        hidden_states = self.apply_weights(hidden_states)

        if self.use_gating:
            gate = torch.sigmoid(self.gate(layer_input))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            hidden_states = hidden_states * gate
        else:
            gate = None
            hidden_states = hidden_states * self.scaling

        return hidden_states, gate


class IA3(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
        name: str = None,
        **kwargs,
    ):
        super().__init__()
        assert config.composition_mode == "scale", "IA3 module only supports composition_mode='scale'."
        assert config.use_dora is False, "IA3 module does not support DoRA."
        if config.r > 1:
            raise ValueError("Can only use composition_mode='scale' when r == 1.")
        self.r = config.r
        self.lora_alpha = config.alpha
        self.composition_mode = config.composition_mode
        self.attn_matrices = config.attn_matrices
        self.use_gating = config.use_gating
        self.name = name
        self.use_dora = config.use_dora
        # Optional dropout
        if config.dropout > 0.0:
            raise ValueError("IA3 module does not support dropout.")

        # Actual trainable parameters
        self.lora_B = nn.Parameter(torch.zeros(lora_B_shape))
        self.scaling = self.lora_alpha

        # Set seed for reproducibility if specified in config
        fix_seed(config.init_weights_seed)

        # For compatibility with LoRA, allow all init_weights types here.
        # Usually should be "ia3".
        if config.init_weights == "lora":
            logger.warning("(IA)^3 module initialized with LoRA zero init. Ignore if this is intended.")
            nn.init.zeros_(self.lora_B)
        elif config.init_weights == "bert":
            nn.init.normal_(self.lora_B, std=0.02)
        elif config.init_weights == "ia3":
            nn.init.ones_(self.lora_B)
        else:
            raise ValueError("Unknown init_weights type: {}".format(config.init_weights))

        if self.use_gating:
            self.gate = nn.Linear(lora_A_shape[-1], gating_heads)
            nn.init.normal_(self.gate.weight, std=0.02)

    @property
    def delta_w(self) -> torch.Tensor:
        return self.lora_B

    def com(self, weights: torch.Tensor, added: torch.Tensor, scaling=None) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if scaling is None:
            scaling = self.scaling
        return weights * (added * scaling)

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        return weights / (added * self.scaling)

    def forward(self, hidden_states: Optional[torch.Tensor], layer_input: torch.Tensor):
        scaling_vector = self.lora_B.view(1, 1, -1).repeat(layer_input.shape[0], 1, 1)
        if hidden_states is None:
            hidden_states = scaling_vector
        else:
            hidden_states = hidden_states * scaling_vector
        if self.use_gating:
            gate = torch.sigmoid(self.gate(layer_input))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            hidden_states = hidden_states * gate
        else:
            gate = None
            hidden_states = hidden_states * self.scaling

        return hidden_states, gate


class Vera(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
        name: str = None,
        **kwargs,
    ):
        super().__init__()
        self.d = config.vera_d
        self.b = config.vera_b
        self.r = config.r
        self.alpha = config.alpha
        self.use_gating = config.use_gating
        self.name = name
        self.dtype = getattr(torch, config.dtype) if config.dtype else None
        fix_seed(config.init_weights_seed)
        self.use_dora = config.use_dora

        # check to make sure that the `composition_mode` is set to `add`
        self.composition_mode = config.composition_mode
        if self.composition_mode != "add":
            raise ValueError("Vera module only supports composition_mode='add'.")

        # Optional dropout
        if config.dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = lambda x: x

        self.lora_A_shape = lora_A_shape
        self.lora_B_shape = lora_B_shape
        self.d_shape = self.lora_A_shape[0]
        self.b_shape = self.lora_B_shape[0]

        # Actual trainable parameters
        self.vera_D = nn.Parameter(torch.diag(torch.ones(self.d_shape) * self.d))
        self.vera_B = nn.Parameter(torch.diag(torch.ones(self.b_shape) * self.b))
        self.scaling = self.alpha / self.r

        if self.use_dora:
            weights = kwargs["w_o"].clone().detach()
            weights = torch.linalg.norm(weights, dim=1).unsqueeze(1).to(dtype=self.dtype)
            self.m = nn.Linear(1, lora_B_shape[0], bias=False, dtype=self.dtype)
            with torch.no_grad():
                self.m.weight.copy_(weights)
        if self.use_gating:
            self.gate = nn.Linear(lora_A_shape[-1], gating_heads)
            nn.init.normal_(self.gate.weight, std=0.02)

    @property
    def delta_w(self) -> torch.Tensor:
        parameters = ForwardContext.get_context().shared_parameters[self.name]
        lora_A = parameters["lora_A"]
        lora_B = parameters["lora_B"]
        return self.vera_B @ lora_B @ self.vera_D @ lora_A

    def com(self, weights: torch.Tensor, added: torch.Tensor, scaling=None) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if scaling is None:
            scaling = self.scaling
        return weights + added * scaling

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        return weights - added * self.scaling

    def forward(self, hidden_states: Optional[torch.Tensor], layer_input: torch.Tensor):
        parameters = ForwardContext.get_context().shared_parameters[self.name]
        lora_A = parameters["lora_A"]
        lora_B = parameters["lora_B"]

        if hidden_states is None:
            hidden_states = layer_input

        if getattr(self, "lora_dropout"):
            hidden_states = self.lora_dropout(hidden_states)

        hidden_states = hidden_states @ torch.t(self.vera_B @ lora_B @ self.vera_D @ lora_A)

        if self.use_gating:
            gate = torch.sigmoid(self.gate(layer_input))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            hidden_states = hidden_states * gate
        else:
            gate = None
            hidden_states = hidden_states * self.scaling

        return hidden_states, gate


def init_shared_vera_parameters(lora_A_shape, lora_B_shape, adapter_config, device):
    """
    This function creates the shared random matrices A and B that are used across all Vera layers.
    These matrices are frozen and initialized according to the specified initialization strategy.

    Args:
        lora_A_shape: The shape of the A matrix
        lora_B_shape: The shape of the B matrix
        adapter_config (dict): The adapter configuration
        device: The device to place the parameters on

    Returns:
        nn.ParameterDict: Dictionary containing:
            - lora_A: Parameter of shape lora_A_shape
            - lora_B: Parameter of shape lora_B_shape
    """
    parameters = nn.ParameterDict()

    # Set seed for reproducibility if specified in config
    fix_seed(adapter_config.init_weights_seed)

    # initialize frozen, random tensors A, B
    dtype = getattr(torch, adapter_config.dtype) if adapter_config.dtype else None
    parameters["lora_A"] = nn.Parameter(torch.zeros(lora_A_shape, dtype=dtype).to(device), requires_grad=False)
    parameters["lora_B"] = nn.Parameter(torch.zeros(lora_B_shape, dtype=dtype).to(device), requires_grad=False)

    if adapter_config["init_weights"] == "lora":
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(parameters["lora_A"], a=math.sqrt(5))
        nn.init.zeros_(parameters["lora_B"])
    elif adapter_config["init_weights"] == "bert":
        nn.init.normal_(parameters["lora_A"], std=0.02)
        nn.init.normal_(parameters["lora_B"], std=0.02)
    elif adapter_config["init_weights"] == "ia3":
        nn.init.ones_(parameters["lora_A"])
        nn.init.ones_(parameters["lora_B"])
    elif adapter_config["init_weights"] == "vera":
        nn.init.kaiming_uniform_(parameters["lora_A"])
        nn.init.kaiming_uniform_(parameters["lora_B"])
    else:
        raise ValueError("Unknown init_weights type: {}".format(adapter_config["init_weights"]))

    return parameters


# DoRA Methods
def compute_dora_norm(weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
    """This function calculates the column-wise norm of the pre-trained weights
    and the LoRA matrices

    eg: for LoRA, it would be ||w_0 + BA||c
    """
    return torch.linalg.norm(weights + added, dim=1)


def compute_dora_deltaw(
    orig_result: torch.Tensor, added: torch.Tensor, m: torch.Tensor, norm: torch.Tensor, scaling: float = 1.0
) -> torch.Tensor:
    """This function calculates the dora update.

    In the paper, the dora update delta_w is calculated as follows:
    m * (w_0x + BAx) / ||w_0 + BA||c
    where:
        - m is the magnitude parameter
        - norm is the column-wise norm i.e ||w_0 + BA||c
        - orig_result is the layer output w_0x
        - added is the LoRA update BAx
    """
    norm_scale = m.weight.view(-1) / norm
    results = norm_scale * (orig_result + added)
    return results


def dora_merge(
    weights: torch.Tensor,
    added: torch.Tensor,
    m: torch.Tensor,
) -> torch.Tensor:
    """This function calculates the weights required to merge with the original pretrained weight matrices."""
    v = weights + added
    norm_scale = m.weight / torch.linalg.norm(v, dim=1).unsqueeze(1)
    result = norm_scale * v

    return result


def compute_dora_add_com_inv(
    weights: torch.Tensor, added: torch.Tensor, m: torch.Tensor, norm: torch.Tensor
) -> torch.Tensor:
    """This function computes the inverse of the DoRA composition operation
    i.e the `compute_dora_deltaw` function with the equation
    w_0' = m * (w_0 + BA) / ||w_0 + BA||c
    """
    result = weights * norm.unsqueeze(1) / m.weight - added
    return result


class LoRALayer(AdapterLayerBase):
    adapter_modules_name = "loras"

    def __init__(
        self,
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shared_parameters = nn.ModuleDict(dict())
        self.location_key = location_key + "_lora"
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.loras = nn.ModuleDict(dict())

        self.merged = False

    def get_n_heads(self, lora: Union[LoRA, IA3, LoRAConfig]):
        return 1

    def _check_lora_location(self, config: LoRAConfig):
        return True

    def _get_lora_shapes(self, config: LoRAConfig):
        raise NotImplementedError()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx

        lora_config = self.adapters_config.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if lora_config is not None and self._check_lora_location(lora_config):
            lora_args = {}
            if lora_config.composition_mode == "add":
                if isinstance(lora_config.vera_d, float) or isinstance(lora_config.vera_b, float):
                    lora_cls = Vera
                    if adapter_name not in self.adapters_config._vera_init_shapes:
                        lora_A_shape, lora_B_shape = self._get_lora_shapes(lora_config)
                        self.adapters_config._vera_init_shapes[adapter_name] = {
                            "lora_A_shape": lora_A_shape,
                            "lora_B_shape": lora_B_shape,
                        }
                else:
                    lora_cls = LoRA
                lora_cls = lora_cls if not isinstance(lora_config, MTLLoRAConfig) else MTLLoRA
            elif lora_config.composition_mode == "scale":
                lora_cls = IA3
            else:
                raise ValueError(f"Unknown composition_mode: {lora_config.composition_mode}")

            # if we're using dora, pass the original weight matrix to initialize the magnitude parameter
            if lora_config.use_dora:
                lora_args["w_o"] = self.weight.data

            lora = lora_cls(
                *self._get_lora_shapes(lora_config),
                config=lora_config,
                gating_heads=self.get_n_heads(lora_config),
                name=adapter_name,
                shared_parameters=self.shared_parameters,
                **lora_args,
            )

            lora.train(self.training)
            lora = lora.to(self.weight.device)
            self.loras[adapter_name] = lora
            return True

        return False

    def share_parameters(
        self,
        name: str,
        adapter_names: List,
        reference_adapter_name: Optional[str] = None,
    ):
        if all(name in self.loras for name in adapter_names):
            shared_params = self.loras[reference_adapter_name].get_parameters()
            self.shared_parameters[name] = shared_params
            for adapter_name in adapter_names:
                self.loras[adapter_name].set_parameters(shared_params)

    def unshare_parameters(self, name: str):
        if name in self.shared_parameters:
            del self.shared_parameters[name]

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.shared_parameters:
            del self.shared_parameters[adapter_name]
        else:
            super().delete_adapter(adapter_name)

    def average_adapter(
        self,
        adapter_name: str,
        input_adapters: Dict[str, float],
        combine_strategy: str,
        svd_rank: int = None,
        **kwargs,
    ) -> bool:
        # add new adapter
        if self.add_adapter(adapter_name, self.layer_idx, **kwargs):
            avg_state_dict = {}

            # First, check if all input adapters are present
            for name in input_adapters.keys():
                if name not in self.loras:
                    self.delete_adapter(adapter_name)  # clean up before raising error
                    raise ValueError("Adapter {} not found.".format(name))

            # VeRA only supports linear averaging.
            if isinstance(self.loras[list(input_adapters.keys())[0]], Vera):
                if combine_strategy != "linear":
                    raise ValueError(
                        "VeRA only supports linear averaging. The combine_strategy must be 'linear'. See https://docs.adapterhub.ml/merging_adapters.html for more information."
                    )

            # Now, combine the weights according to the strategy
            if combine_strategy == "linear":
                for name, weight in input_adapters.items():
                    module = self.loras[name]
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v

            elif combine_strategy == "lora_linear_only_negate_b":
                # Same as linear but for negative weights only negate the B matrix and leave A positive
                # See Zhang et al. (2023) https://proceedings.neurips.cc/paper_files/paper/2023/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html
                for name, weight in input_adapters.items():
                    module = self.loras[name]
                    for k, v in module.state_dict().items():
                        if "lora_B" in k:
                            zhang_weight = weight
                        elif "lora_A" in k:
                            zhang_weight = abs(weight)
                        else:
                            # This should never happen as we only have lora_A and lora_B in the state_dict
                            raise ValueError(
                                f"Key must either contain 'lora_A' or 'lora_B' but is {k}. This should never"
                                " happen. Please open an issue on GitHub if you encounter this error."
                            )

                        if k in avg_state_dict:
                            avg_state_dict[k] += zhang_weight * v
                        else:
                            avg_state_dict[k] = zhang_weight * v

            elif combine_strategy == "lora_delta_w_svd":
                # Weight the delta_w matrices by the input weights and then use Singular Value Decomposition (SVD) to split them into A and B matrices.
                self._average_adapter_lora_delta_w_svd(input_adapters, avg_state_dict, svd_rank)

            else:
                raise ValueError(f"The combine_strategy '{combine_strategy}' is not supported for LoRA.")

            # load averaged weights
            self.loras[adapter_name].load_state_dict(avg_state_dict)
            return True

        return False

    def get_adapter(self, adapter_name: str) -> nn.Module:
        """Returns the adapter module with the given name.

        Args:
            adapter_name (str): The name of the adapter module.
        """
        if adapter_name in self.adapter_modules:
            return self.adapter_modules[adapter_name]
        elif adapter_name in self.shared_parameters:
            return self.shared_parameters[adapter_name]
        else:
            return None

    def _average_adapter_lora_delta_w_svd(self, input_adapters: Dict[str, float], avg_state_dict, svd_rank):
        # Weight the delta_w matrices by the input weights and then use Singular Value Decomposition to split them into A and B matrices.
        if svd_rank is None:
            raise ValueError("svd_rank must be set when using 'lora_delta_w_svd'.")

        # Collect delta_w matrices. Shape of every delta_w matrix in the list: d×k
        delta_w = [self.loras[adapter_name].delta_w for adapter_name in input_adapters.keys()]

        # If the lora has fan_in_fan_out, we need to transpose the matrices
        if self.fan_in_fan_out:
            delta_w = [torch.t(delta_w) for delta_w in delta_w]

        delta_w = torch.stack(delta_w, dim=0)  # shape: n×d×k

        # Weight the delta_w matrices
        weights = torch.tensor(list(input_adapters.values()), device=delta_w.device)  # shape: n
        weights = weights.view(-1, 1, 1)  # shape: n×1×1
        delta_w = delta_w * weights  # shape: n×d×k

        # Now bring down to d×k matrix
        delta_w = delta_w.sum(dim=0)  # shape: d×k

        # Perform SVD to split delta_w into A and B matrices
        U, S_diag, V = torch.linalg.svd(delta_w)

        # Reduce rank
        U = U[:, :svd_rank]  # U is 2D
        S_diag = S_diag[:svd_rank]  # S_diag is 1D
        V = V[:svd_rank, :]  # V is 2D

        # The SVD has decomposed delta_w into U, S, and V such that: delta_w = U @ S_diag @ V
        # In LoRA we have: delta_w = B @ A
        # Hence, we can set: A = V and B = U @ S_diag
        if self.fan_in_fan_out:
            avg_state_dict["lora_A"] = torch.t(V)
            avg_state_dict["lora_B"] = torch.t(U @ torch.diag(S_diag))
        else:
            avg_state_dict["lora_A"] = V
            avg_state_dict["lora_B"] = U @ torch.diag(S_diag)

    def _copy_hooks_from(self, module: nn.Module):
        for (
            k,
            v,
        ) in module.__dict__.items():
            if "_hooks" in k:
                setattr(self, k, v)


class LoRAState(NamedTuple):
    """Models the input and output states of a LoRA layer.

    Args:
        layer_input (torch.Tensor): The input states to the adapted layer.
        hidden_states (Optional[torch.Tensor]):
            The hidden states of the adaptation module. These can be None before passing through the first LoRA/ IA3
            module.
        layer_output (torch.Tensor): The output states of the original layer without adaptation.
        last (str, optional): Name of the last adapter applied in the composition.
    """

    layer_input: torch.Tensor
    hidden_states: Optional[torch.Tensor]
    layer_output: torch.Tensor
    last: Optional[str]


class LoRALinear(LoRALayer, ComposableAdapterLayerBase):
    """
    LoRA implementation for Linear layer. This layer supports composition.

    Args:
        fan_in_fan_out (bool, optional):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). Defaults to False.
        no_init_bias (bool, optional): Use this to add a bias that is not initialized by PyTorch. Defaults to False.

    """

    supported_compositions = [
        Stack,
        BatchSplit,
        Average,
        Parallel,
        MultiTask,
    ]
    allow_multi_parallelize = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        attn_key: str = None,
        fan_in_fan_out: bool = False,
        no_init_bias: bool = False,
        **kwargs,
    ):
        if no_init_bias and "bias" not in kwargs:
            kwargs["bias"] = False
        LoRALayer.__init__(
            self,
            location_key,
            model_config,
            adapters_config,
            in_features,
            out_features,
            **kwargs,
        )

        self.attn_key = attn_key
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = torch.t(self.weight.data)
        if no_init_bias:
            self.bias = nn.Parameter(torch.empty(out_features))

    @classmethod
    def wrap(
        cls,
        module: Union[nn.Linear, Conv1D],
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        attn_key: str = None,
        **kwargs,
    ):
        if isinstance(module, Conv1D):
            new_module = LoRALinearTorch(
                module.weight.shape[0],
                module.weight.shape[1],
                location_key,
                model_config,
                adapters_config,
                attn_key=attn_key,
                **kwargs,
            )
        else:
            if bitsandbytes_available and isinstance(module, Linear4bit):
                cls = LoRALinear4bit
            elif bitsandbytes_available and isinstance(module, Linear8bitLt):
                cls = LoRALinear8bitLt
            else:
                cls = LoRALinearTorch
            # Make sure that the bias is not added if the original module does not have one
            if "bias" not in kwargs:
                kwargs["bias"] = hasattr(module, "bias") and module.bias is not None
            new_module = cls(
                module.in_features,
                module.out_features,
                location_key,
                model_config,
                adapters_config,
                attn_key=attn_key,
                **kwargs,
            )
        new_module.copy_from(module)
        new_module._copy_hooks_from(module)

        return new_module

    def copy_from(self, module: nn.Linear):
        self.weight = module.weight
        if module.bias is not None:
            self.bias = module.bias

    def _check_lora_location(self, config: LoRAConfig):
        return self.attn_key is None or self.attn_key in config.attn_matrices

    def _get_lora_shapes(self, config: LoRAConfig):
        return (config.r, self.in_features), (self.out_features, config.r)

    def maybe_t(self, w):
        return torch.t(w) if self.fan_in_fan_out else w

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                delta_w = self.maybe_t(lora.delta_w)
                # we check if the lora module has 'm', if so then we're using dora
                if lora.use_dora:
                    setattr(lora, "dora_w_o", self.weight.data)
                    self.weight.data = dora_merge(self.weight.data, delta_w, lora.m)
                else:
                    self.weight.data = lora.com(self.weight.data, delta_w)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

    def reset_adapter(self):
        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            delta_w = self.maybe_t(lora.delta_w)

            # we check if the lora module has 'm', if so then we're using dora
            if lora.use_dora:
                w_o = getattr(lora, "dora_w_o", None)
                norm = compute_dora_norm(w_o, delta_w)
                delta_w = compute_dora_add_com_inv(self.weight.data, delta_w, lora.m, norm)
                # remove the dora_w_o attribute we set in the merge_adapter method
                del lora.dora_w_o
            else:
                self.weight.data = lora.com_inv(self.weight.data, delta_w)
            self.merged = None

    def vslice(self, state: LoRAState, slice_obj: slice) -> LoRAState:
        return LoRAState(
            state.layer_input[slice_obj],
            (state.hidden_states[slice_obj] if state.hidden_states is not None else None),
            state.layer_output[slice_obj],
            state.last,
        )

    def pad_and_concat(self, states: List[LoRAState]) -> LoRAState:
        return LoRAState(
            torch.cat([s.layer_input for s in states], dim=0),
            (torch.cat([s.hidden_states for s in states], dim=0) if states[0].hidden_states is not None else None),
            torch.cat([s.layer_output for s in states], dim=0),
            states[-1].last,
        )

    def repeat(self, state: LoRAState, channels: int) -> LoRAState:
        return LoRAState(
            state.layer_input.repeat(channels, 1, 1),
            (state.hidden_states.repeat(channels, 1, 1) if state.hidden_states is not None else None),
            state.layer_output.repeat(channels, 1, 1),
            state.last,
        )

    def mean(self, states: List[LoRAState], weights: torch.Tensor) -> LoRAState:
        return LoRAState(
            states[0].layer_input,
            (
                torch.mean(
                    torch.stack([s.hidden_states for s in states], dim=0) * weights,
                    dim=0,
                )
                if states[0].hidden_states is not None
                else None
            ),
            states[0].layer_output,
            states[-1].last,
        )

    def compose_single(self, adapter_setup: str, state: LoRAState, lvl: int = 0) -> LoRAState:
        lora = self.loras[adapter_setup]
        hidden_states, gate = lora(state.hidden_states, state.layer_input)
        if gate is not None:
            self._store_gating_score(adapter_setup, gate)

        return state._replace(hidden_states=hidden_states, last=adapter_setup)

    def forward(self, input_states: torch.Tensor):
        if self.fan_in_fan_out:
            weight = torch.transpose(self.weight, -2, -1) if self.fan_in_fan_out else self.weight
            # result shape: <batch_size> x <seq_len> x <head_dim>
            layer_output = F.linear(input_states, weight, bias=self.bias)
        else:
            layer_output = super().forward(input_states)

        if not self.merged:
            adapter_setup = self.get_active_setup()
            if adapter_setup is not None:
                state = LoRAState(input_states, None, layer_output, None)
                state = self.compose(adapter_setup, state)
                _, hidden_states, layer_output, last = state

                last_lora = self.loras[last]

                # we check if the last_lora module has 'm', if so then we're using dora
                if last_lora.use_dora:
                    norm = compute_dora_norm(self.weight, last_lora.delta_w)
                    layer_output = compute_dora_deltaw(layer_output, hidden_states, last_lora.m, norm)
                else:
                    layer_output = last_lora.com(
                        layer_output, hidden_states, scaling=1.0
                    )  # scaling already applied in compose

        return layer_output


class LoRALinearTorch(LoRALinear, nn.Linear):
    pass


if bitsandbytes_available:

    class LoRALinear4bit(LoRALinear, Linear4bit):
        def copy_from(self, module: Linear4bit):
            self.weight = module.weight
            if module.bias is not None:
                self.bias = module.bias
            self.compute_dtype = module.compute_dtype
            self.compute_type_is_set = module.compute_type_is_set
            self.quant_state = module.quant_state
            self.quant_storage = module.quant_storage

        def merge_adapter(self, name: str):
            if name in self.loras:
                if self.merged == name:
                    return  # already merged
                elif not self.merged:
                    lora = self.loras[name]
                    if lora.use_gating:
                        raise ValueError("Cannot merge LoRA layer with gating.")
                    delta_w = self.maybe_t(lora.delta_w)
                    layer_weight = dequantize_bnb_weight(self.weight, state=self.quant_state)
                    kwargs = self.weight.__dict__
                    merged_weight = lora.com(layer_weight, delta_w)
                    self.weight = Params4bit(merged_weight.to("cpu"), requires_grad=False, **kwargs).to(
                        self.weight.device
                    )
                    self.merged = name
                elif self.merged != name:
                    raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

        def reset_adapter(self):
            if self.merged:
                lora = self.loras[self.merged]
                delta_w = self.maybe_t(lora.delta_w)
                merged_weight = dequantize_bnb_weight(self.weight, state=self.quant_state)
                kwargs = self.weight.__dict__
                layer_weight = lora.com_inv(merged_weight, delta_w)
                self.weight = Params4bit(layer_weight.to("cpu"), requires_grad=False, **kwargs).to(self.weight.device)
                self.merged = None

    class LoRALinear8bitLt(LoRALinear, Linear8bitLt):
        def copy_from(self, module: Linear8bitLt):
            self.weight = module.weight
            if module.bias is not None:
                self.bias = module.bias
            self.state = module.state
            self.index = module.index

        def merge_adapter(self, name: str):
            if name in self.loras:
                if self.merged == name:
                    return  # already merged
                elif not self.merged:
                    lora = self.loras[name]
                    if lora.use_gating:
                        raise ValueError("Cannot merge LoRA layer with gating.")
                    delta_w = self.maybe_t(lora.delta_w)
                    layer_weight = dequantize_bnb_weight(self.weight, state=self.state)
                    merged_weight = lora.com(layer_weight, delta_w)
                    self.weight = Int8Params(
                        merged_weight.to("cpu"),
                        requires_grad=False,
                        has_fp16_weights=self.weight.has_fp16_weights,
                    ).to(self.weight.device)
                    self.state.reset_grads()
                    self.merged = name
                elif self.merged != name:
                    raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

        def reset_adapter(self):
            if self.merged:
                lora = self.loras[self.merged]
                delta_w = self.maybe_t(lora.delta_w)
                merged_weight = dequantize_bnb_weight(self.weight, state=self.state)
                layer_weight = lora.com_inv(merged_weight, delta_w)
                self.weight = Int8Params(
                    layer_weight.to("cpu"),
                    requires_grad=False,
                    has_fp16_weights=self.weight.has_fp16_weights,
                ).to(self.weight.device)
                self.state.reset_grads()
                self.merged = None


class LoRAMergedLinear(LoRALayer, nn.Linear):
    """
    LoRA implementation for merged attention layer, as used by some model implementations (e.g. GPT-2). This layer
    currently does not support composition.

    Args:
        fan_in_fan_out (bool, optional):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). Defaults to False.
        no_init_bias (bool, optional): Use this to add a bias that is not initialized by PyTorch. Defaults to False.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        fan_in_fan_out: bool = False,
        no_init_bias: bool = False,
        **kwargs,
    ):
        if no_init_bias and "bias" not in kwargs:
            kwargs["bias"] = False
        LoRALayer.__init__(
            self,
            location_key,
            model_config,
            adapters_config,
            in_features,
            out_features,
            **kwargs,
        )

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        if no_init_bias:
            self.bias = nn.Parameter(torch.empty(out_features))

    @classmethod
    def wrap(
        cls,
        module: Union[nn.Linear, Conv1D],
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        **kwargs,
    ):
        if isinstance(module, Conv1D):
            new_module = cls(
                module.weight.shape[0],
                module.weight.shape[1],
                location_key,
                model_config,
                adapters_config,
                **kwargs,
            )
        else:
            new_module = cls(
                module.in_features,
                module.out_features,
                location_key,
                model_config,
                adapters_config,
                **kwargs,
            )
        new_module.weight = module.weight
        if module.bias is not None:
            new_module.bias = module.bias
        new_module._copy_hooks_from(module)

        return new_module

    def get_n_heads(self, lora: Union[LoRA, IA3, LoRAConfig]):
        return len(set(lora.attn_matrices))

    def _get_lora_shapes(self, config: LoRAConfig):
        n_heads = self.get_n_heads(config)
        return (config.r * n_heads, self.in_features), (
            self.out_features // 3 * n_heads,
            config.r,
        )

    def add_adapter(self, adapter_name: str, layer_idx: int, **kwargs) -> bool:
        is_added = super().add_adapter(adapter_name, layer_idx, **kwargs)
        if is_added:
            lora_config = lora_config = self.adapters_config.match(
                adapter_name,
                config_type=LoRAConfig,
                layer_idx=self.layer_idx,
                location_key=self.location_key,
            )
            lora = self.loras[adapter_name]
            lora.enable_lora = [
                "q" in lora_config.attn_matrices,
                "k" in lora_config.attn_matrices,
                "v" in lora_config.attn_matrices,
            ]
            # Actual trainable parameters
            if any(lora.enable_lora):
                # Compute the indices
                # Ensure we're not creating a meta tensor
                device = self.weight.device if self.weight.device.type != "meta" else torch.device("cpu")
                lora_ind = torch.zeros((self.out_features,), dtype=torch.bool, device=device).view(
                    len(lora.enable_lora), -1
                )
                lora_ind[lora.enable_lora, :] = True
                lora.lora_ind = lora_ind.view(-1)
            return True
        else:
            return False

    def pad(self, x, lora, fill_value=None):
        if fill_value is None:
            if lora.composition_mode == "add":
                fill_value = 0
            else:
                fill_value = 1
        result = x.new_full((*x.shape[:-1], self.out_features), fill_value)
        result = result.view(-1, self.out_features)
        # Move lora_ind to the same device as x
        lora_ind = lora.lora_ind.to(x.device)
        result[:, lora_ind] = x.reshape(-1, self.out_features // 3 * self.get_n_heads(lora))
        return result.view((*x.shape[:-1], self.out_features))

    def reset_adapter(self):
        def T(w):
            return w if self.fan_in_fan_out else torch.t(w)

        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0 and any(lora.enable_lora):
                if lora.composition_mode == "scale":
                    delta_w = lora.lora_B
                else:
                    delta_w = F.conv1d(
                        lora.lora_A.data.unsqueeze(0),
                        lora.lora_B.data.unsqueeze(-1),
                        groups=sum(lora.enable_lora),
                    ).squeeze(0)
                # shape after transpose: <head_dim> x <head_dim * n_heads>
                delta_w = delta_w.transpose(-2, -1)
                self.weight.data = lora.com_inv(self.weight.data, T(self.pad(delta_w, lora)))
            self.merged = None

    def _compute_adapted_weight(self, name, lora):
        def T(w):
            return w if self.fan_in_fan_out else torch.t(w)

        weight = self.weight
        if lora.r > 0:
            if lora.composition_mode == "scale":
                delta_w = lora.lora_B
            else:
                delta_w = F.conv1d(
                    lora.lora_A.data.unsqueeze(0),
                    lora.lora_B.data.unsqueeze(-1),
                    groups=sum(lora.enable_lora),
                ).squeeze(0)
            # shape after transpose: <head_dim> x <head_dim * n_heads>
            delta_w = delta_w.transpose(-2, -1)
            weight = lora.com(weight, T(self.pad(delta_w, lora)))

        return weight

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                self.weight.data = self._compute_adapted_weight(name, lora)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w

        if not self.merged:
            adapter_setup = self.get_active_setup()
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    lora = self.loras[adapter_setup[0]]
                    if lora.r > 0:
                        if lora.composition_mode == "scale":
                            delta_w = lora.lora_B.view(1, 1, -1)
                        else:
                            after_A = F.linear(lora.lora_dropout(x), lora.lora_A)
                            after_B = F.conv1d(
                                after_A.transpose(-2, -1),
                                lora.lora_B.unsqueeze(-1),
                                groups=sum(lora.enable_lora),
                            ).transpose(-2, -1)
                            delta_w = after_B
                        if lora.use_gating:
                            gate = torch.sigmoid(lora.gate(x))
                            gate = torch.mean(gate, dim=1)
                            self._store_gating_score(adapter_setup[0], gate)
                            gate = self.pad(
                                gate.repeat_interleave(self.out_features // 3, dim=-1),
                                lora,
                                fill_value=1,
                            ).unsqueeze(1)
                        else:
                            gate = None
                        # result = (batch_size, seq_len, head_dim * 3)
                        result = lora.com(result, self.pad(delta_w, lora), scaling=gate)
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")

        return F.linear(x, T(self.weight), bias=self.bias)


def init_lora(model):
    model = model.base_model

    # First, check if we should use merged linear layer instead of individual q,k,v projections
    use_merged_linear = model.adapter_interface.attn_qkv_proj is not None

    for _, _, attention in model.iter_attentions():
        if use_merged_linear:
            if qkv_proj := multigetattr(attention, model.adapter_interface.attn_qkv_proj, None):
                lora_proj = LoRAMergedLinear.wrap(qkv_proj, "selfattn", model.config, model.adapters_config)
                multisetattr(attention, model.adapter_interface.attn_qkv_proj, lora_proj)
        else:
            if q_proj := multigetattr(attention, model.adapter_interface.attn_q_proj, None):
                lora_proj = LoRALinear.wrap(
                    q_proj,
                    "selfattn",
                    model.config,
                    model.adapters_config,
                    attn_key="q",
                )
                multisetattr(attention, model.adapter_interface.attn_q_proj, lora_proj)
            if k_proj := multigetattr(attention, model.adapter_interface.attn_k_proj, None):
                lora_proj = LoRALinear.wrap(
                    k_proj,
                    "selfattn",
                    model.config,
                    model.adapters_config,
                    attn_key="k",
                )
                multisetattr(attention, model.adapter_interface.attn_k_proj, lora_proj)
            if v_proj := multigetattr(attention, model.adapter_interface.attn_v_proj, None):
                lora_proj = LoRALinear.wrap(
                    v_proj,
                    "selfattn",
                    model.config,
                    model.adapters_config,
                    attn_key="v",
                )
                multisetattr(attention, model.adapter_interface.attn_v_proj, lora_proj)

    for _, layer in model.iter_layers():
        if intermediate_proj := multigetattr(layer, model.adapter_interface.layer_intermediate_proj):
            lora_proj = LoRALinear.wrap(
                intermediate_proj,
                "intermediate",
                model.config,
                model.adapters_config,
            )
            multisetattr(
                layer,
                model.adapter_interface.layer_intermediate_proj,
                lora_proj,
            )
        if output_proj := multigetattr(layer, model.adapter_interface.layer_output_proj):
            lora_proj = LoRALinear.wrap(output_proj, "output", model.config, model.adapters_config)
            multisetattr(layer, model.adapter_interface.layer_output_proj, lora_proj)
