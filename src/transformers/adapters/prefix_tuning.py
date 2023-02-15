from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..modeling_utils import ModuleUtilsMixin
from .composition import AdapterCompositionBlock, BatchSplit, Parallel, Stack, adjust_tensors_for_parallel
from .configuration import PrefixTuningConfig
from .context import AdapterSetup, ForwardContext
from .layer import AdapterLayerBase
from .modeling import Activation_Function_Class


class PrefixTuning(nn.Module, ModuleUtilsMixin):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        config: PrefixTuningConfig,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = self.input_size // self.n_heads
        self.config = config

        self.wte = nn.Embedding(self.config.prefix_length, self.input_size)
        self.control_trans = nn.Sequential(
            nn.Linear(self.input_size, self.config.bottleneck_size),
            Activation_Function_Class(self.config.non_linearity.lower()),
            nn.Linear(self.config.bottleneck_size, self.n_layers * 2 * self.input_size),
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def eject(self):
        input_tokens = torch.arange(self.config.prefix_length).long()
        input_tokens = input_tokens.unsqueeze(0).expand(1, -1).to(self.device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            self.config.prefix_length * self.n_layers * 2 * self.input_size
        )  # *2 for key and value

        return key_values

    def forward(self, batch_size):
        input_tokens = torch.arange(self.config.prefix_length).long()
        input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            batch_size, self.config.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


class FlatPrefixTuning(nn.Module, ModuleUtilsMixin):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        config: PrefixTuningConfig,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = self.input_size // self.n_heads
        self.config = config

        self.control_trans = nn.Parameter(torch.randn(self.config.prefix_length * self.n_layers * 2 * self.input_size))

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, batch_size):
        key_values = (
            self.control_trans.unsqueeze(0)
            .expand(batch_size, -1)
            .view(batch_size, self.config.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head)
            .to(self.device)
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


class PrefixTuningGroup(nn.ModuleDict):
    def __init__(self, module_configs, prefix_tuning_config):
        super().__init__()
        if prefix_tuning_config["flat"]:
            prefix_tuning_class = FlatPrefixTuning
        else:
            prefix_tuning_class = PrefixTuning
        for k, kwargs in module_configs.items():
            self[k] = prefix_tuning_class(**kwargs, config=prefix_tuning_config)

    def eject(self):
        """Converts all PrefixTuning modules into FlatPrefixTuning modules."""
        for k, v in self.items():
            if isinstance(v, PrefixTuning):
                config = v.config.replace(flat=True)
                self[k] = FlatPrefixTuning(v.n_layers, v.n_heads, v.input_size, config)
                weights = v.eject()
                self[k].control_trans = nn.Parameter(weights)

    def forward(self, batch_size):
        return {k: v(batch_size) for k, v in self.items()}


class PrefixTuningPool(nn.Module):
    """
    The model layer that holds all Prefix Tuning prefixes. While each Transformers layer has its own prefix, this layer
    is shared across all Transformers layers.

    How it works:

        1. A `PrefixTuningShim` module that sets this module as pool module is added to each layer.
        2. On adding a prefix, each shim module where a prefix should be added increments a counter in `prefix_counts`.
        3. Finally, the base model class confirms adding a new prefix by calling `confirm_prefix()`.
        4. This module adds a prefix layer that produces outputs corresponding to the indicated number of layers.

    Notes:

        - The forward call to this layer is executed in the ForwardContext of each model pass.
        - All other methods of this class (except for `confirm_prefix()`) should be called exclusively by
          `PrefixTuningShim`.

    Args:
        config (:class:`~transformers.PretrainedConfig`): The model config.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prefix_counts = {}
        self.prefix_tunings = nn.ModuleDict()

    def indicate_prefix(self, prefix_name: str, location_key: str, **kwargs):
        if prefix_name not in self.prefix_counts:
            self.prefix_counts[prefix_name] = {location_key: {"count": 1, **kwargs}}
        elif location_key not in self.prefix_counts[prefix_name]:
            self.prefix_counts[prefix_name][location_key] = {"count": 1, **kwargs}
        else:
            # TODO-AH: Check if kwargs are the same
            self.prefix_counts[prefix_name][location_key]["count"] += 1

        return self.prefix_counts[prefix_name][location_key]["count"] - 1

    def confirm_prefix(self, prefix_name: str):
        """Create Prefix Tuning module based on shim layer infications."""
        prefix_tuning_config = self.config.adapters.match(prefix_name, PrefixTuningConfig)
        if prefix_tuning_config is None:
            return

        if prefix_name not in self.prefix_counts:
            raise ValueError(f"Prefix {prefix_name} not found in PrefixTuningPool")

        module_configs = {}
        for location_key, location_config in self.prefix_counts[prefix_name].items():
            module_configs[location_key] = {
                "n_layers": location_config["count"],
                "n_heads": location_config["n_heads"],
                "input_size": location_config["input_size"],
            }
        prefix_tuning = PrefixTuningGroup(module_configs, prefix_tuning_config)
        prefix_tuning.train(self.training)  # make sure training mode is consistent
        self.prefix_tunings[prefix_name] = prefix_tuning
        del self.prefix_counts[prefix_name]

    def delete_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            del self.prefix_tunings[prefix_name]

    def enable_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            for param in self.prefix_tunings[prefix_name].parameters():
                param.requires_grad = True

    def get_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            return self.prefix_tunings[prefix_name]
        else:
            return None

    def forward(self, *args, **kwargs):
        context = AdapterSetup.get_context()
        if context is not None:
            adapter_setup = context.adapter_setup
        else:
            adapter_setup = self.config.adapters.active_setup

        prefix_states = {}
        if adapter_setup is not None:
            # Infer batch size
            input_tensor_names = ["input_ids", "decoder_input_ids", "attention_mask", "inputs_embeds", "pixel_values"]
            batch_size = None
            for name in input_tensor_names:
                if kwargs.get(name, None) is not None:
                    batch_size = kwargs[name].size(0)
                    break
            if batch_size is None:
                if len(args) > 0:
                    batch_size = args[0].size(0)
                else:
                    raise ValueError("Could not infer batch size for prefix tuning from inputs.")

            # Pass to sub-layers
            for name in adapter_setup.flatten():
                if name in self.prefix_tunings:
                    prefix_states[name] = self.prefix_tunings[name](batch_size)

        return prefix_states


class PrefixTuningShim(AdapterLayerBase, nn.Module):
    """
    Representation of a Prefix Tuning layer within one Transformer layer. This class implements `AdapterLayerBase` for
    compatibility with adapters. It uses `PrefixTuningPool` in the background and `set_pool()` must be called after
    initialization.

    Args:
        location_key (str): The id describing the location of this layer in the model.
                            Currently, can be "encoder_prefix", "cross_prefix" or None.
        config (:class:`~transformers.PretrainedConfig`): The model config.
    """

    def __init__(self, location_key: str, config, add_model_type_to_key: bool = False):
        super().__init__()
        self.config = config
        self.location_key = location_key
        if add_model_type_to_key:
            self.location_key = f"{self.config.model_type}_{self.location_key}"
        self.prefixes = {}
        self.prefix_gates = nn.ModuleDict()

    def set_pool(self, pool: PrefixTuningPool):
        self.__setattr__("pool", pool)

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
            prefix_id = self.pool.indicate_prefix(
                adapter_name,
                self.location_key,
                n_heads=self.config.num_attention_heads,
                input_size=self.config.hidden_size,
            )
            self.prefixes[adapter_name] = prefix_id

            if prefix_tuning_config.use_gating:
                gate_outputs = 1 if prefix_tuning_config.shared_gating else 2
                gate = nn.Linear(self.config.hidden_size, gate_outputs)
                gate.weight.data.normal_(mean=0.0, std=0.02)
                self.prefix_gates[adapter_name] = gate

    def delete_adapter(self, adapter_name: str):
        self.pool.delete_prefix(adapter_name)
        if adapter_name in self.prefixes:
            del self.prefixes[adapter_name]
        if adapter_name in self.prefix_gates:
            del self.prefix_gates[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to prefix tuning

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to prefix tuning

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for prefix_tuning_name in adapter_setup.flatten():
                self.pool.enable_prefix(prefix_tuning_name)
                if prefix_tuning_name in self.prefix_gates:
                    for param in self.prefix_gates[prefix_tuning_name].parameters():
                        param.requires_grad = unfreeze_adapters

    def get_adapter(self, adapter_name):
        return_dict = nn.ModuleDict()
        # Make sure to only return params once
        if adapter_name in self.prefixes and self.prefixes[adapter_name] == 0:
            prefix_module = self.pool.get_prefix(adapter_name)
            if prefix_module is not None:
                return_dict["prefix"] = prefix_module[self.location_key]
        if adapter_name in self.prefix_gates:
            return_dict["gate"] = self.prefix_gates[adapter_name]
        if len(return_dict) > 0:
            return return_dict

        return None

    def single_forward(
        self,
        adapter_name: str,
        key_states,
        value_states,
        residual_input,
        attention_mask=None,
        invert_mask=True,
        idx_range=None,
    ):
        prefix_id = self.prefixes[adapter_name]
        batch_size = key_states.size(0)

        # Retrieve pre-computed prefix states from context
        context = ForwardContext.get_context()
        # batch_size x n_heads x prefix_length x n_embd_per_head
        prefix_keys, prefix_values = context.prefix_states[adapter_name][self.location_key][prefix_id]

        # select index range for batch split
        if idx_range is not None:
            prefix_keys = prefix_keys[idx_range]
            prefix_values = prefix_values[idx_range]

        if adapter_name in self.prefix_gates:
            gate = self.prefix_gates[adapter_name]
            gate_output = torch.mean(torch.sigmoid(gate(residual_input)), dim=1)
            self._store_gating_score(adapter_name, gate_output)
            gate_output_key = gate_output[:, 0].view(-1, 1, 1, 1)
            gate_output_value = gate_output[:, -1].view(-1, 1, 1, 1)
            prefix_keys = prefix_keys * gate_output_key
            prefix_values = prefix_values * gate_output_value

        # replicate for Parallel block
        prefix_keys, prefix_values = adjust_tensors_for_parallel(key_states, prefix_keys, prefix_values)

        key_states = torch.cat([prefix_keys, key_states], dim=2)
        value_states = torch.cat([prefix_values, value_states], dim=2)
        if attention_mask is not None:
            if attention_mask.dim() == 2:  # e.g. for DistilBERT, attention_mask has shape (batch_size, seq_len)
                prefix_mask = torch.ones(batch_size, prefix_keys.size(2)).to(attention_mask.device)
            else:
                prefix_mask = torch.ones(batch_size, 1, attention_mask.size(2), prefix_keys.size(2)).to(
                    attention_mask.device
                )
            if invert_mask:
                prefix_mask = 1.0 - prefix_mask
            (prefix_mask,) = adjust_tensors_for_parallel(attention_mask, prefix_mask)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)

        return key_states, value_states, residual_input, attention_mask

    def _pad_and_concat(self, max_prefix_length, outputs, invert_mask=True):
        """Pads all key & value states to the lFongest prefix length in the current batch.
        This is required e.g. for stacked prefix tunings.
        """
        all_key_states, all_value_states, all_residual_input, all_attention_mask = [], [], [], []
        for key_states, value_states, residual_input, attention_mask in outputs:
            # pad sizes
            pad_length = max_prefix_length - key_states.shape[-2]
            pad_size = (0, 0, pad_length, 0)
            key_states = F.pad(key_states, pad_size, "constant", self.config.pad_token_id)
            value_states = F.pad(value_states, pad_size, "constant", self.config.pad_token_id)

            # pad attention mask
            if pad_length > 0:
                # Masking the padded tokens only works correctly if attention_mask is set
                # We assume this to be the case at this point
                assert attention_mask is not None, "Attention mask must be set for prefix tuning"
                attention_mask = F.pad(
                    attention_mask,
                    (max_prefix_length - attention_mask.shape[-1], 0),
                    "constant",
                    1.0 if invert_mask else 0.0,
                )

            all_key_states.append(key_states)
            all_value_states.append(value_states)
            all_residual_input.append(residual_input)
            all_attention_mask.append(attention_mask)

        all_key_states = torch.cat(all_key_states, dim=0)
        all_value_states = torch.cat(all_value_states, dim=0)
        all_residual_input = torch.cat(all_residual_input, dim=0)
        all_attention_mask = torch.cat(all_attention_mask, dim=0) if attention_mask is not None else None

        return all_key_states, all_value_states, all_residual_input, all_attention_mask

    def adapter_stack(
        self,
        adapter_setup: Stack,
        key_states,
        value_states,
        residual_input,
        attention_mask=None,
        invert_mask=True,
        idx_range=None,
        lvl=0,
    ):
        for adapter_stack_layer in adapter_setup:
            # Break if setup is too deep
            if isinstance(adapter_stack_layer, AdapterCompositionBlock) and lvl >= 1:
                raise ValueError(
                    "Specified adapter setup is too deep. Cannot have {} at level {}".format(
                        adapter_stack_layer.__class__.__name__, lvl
                    )
                )
            # We have a nested parallel layer -> call parallel method
            elif isinstance(adapter_stack_layer, Parallel):
                key_states, value_states, residual_input, attention_mask = self.adapter_parallel(
                    adapter_stack_layer,
                    key_states,
                    value_states,
                    residual_input,
                    attention_mask,
                    invert_mask=invert_mask,
                    idx_range=idx_range,
                    lvl=lvl + 1,
                )
            # We have a nested batch split block -> call batchsplit method
            elif isinstance(adapter_stack_layer, BatchSplit):
                key_states, value_states, residual_input, attention_mask = self.adapter_batchsplit(
                    adapter_stack_layer,
                    key_states,
                    value_states,
                    residual_input,
                    attention_mask,
                    invert_mask=invert_mask,
                    idx_range=idx_range,
                    lvl=lvl + 1,
                )
            # We have a single prefix tuning module part of this model -> forward pass
            elif adapter_stack_layer in self.prefixes:
                key_states, value_states, _, attention_mask = self.single_forward(
                    adapter_stack_layer,
                    key_states,
                    value_states,
                    residual_input,
                    attention_mask,
                    invert_mask,
                    idx_range=idx_range,
                )
            # Nesting other composition blocks is invalid
            elif isinstance(adapter_stack_layer, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_stack_layer.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # As all prefix tuning modules are centrally stored, fail if not found.
            else:
                raise ValueError(f"Unknown prefix tuning name '{adapter_stack_layer}'.")

        return key_states, value_states, residual_input, attention_mask

    def adapter_parallel(
        self,
        adapter_setup: Parallel,
        key_states,
        value_states,
        residual_input,
        attention_mask=None,
        invert_mask=True,
        idx_range=None,
        lvl=0,
    ):
        """
        For parallel execution of the adapters on the same input. This means that the input is repeated N times before
        feeding it to the adapters (where N is the number of adapters).
        """

        context = ForwardContext.get_context()
        if not context.adapters_parallelized:
            orig_batch_size = residual_input.shape[0]
            residual_input = residual_input.repeat(self.config.adapters.active_setup.parallel_channels, 1, 1, 1)
            key_states = key_states.repeat(self.config.adapters.active_setup.parallel_channels, 1, 1, 1)
            value_states = value_states.repeat(self.config.adapters.active_setup.parallel_channels, 1, 1, 1)
            if attention_mask is not None:
                if attention_mask.dim() == 2:  # e.g. for DistilBERT, attention_mask has shape (batch_size, seq_len)
                    attention_mask = attention_mask.repeat(self.config.adapters.active_setup.parallel_channels, 1)
                else:
                    attention_mask = attention_mask.repeat(
                        self.config.adapters.active_setup.parallel_channels, 1, 1, 1
                    )
            context.adapters_parallelized = True
        else:
            # The base model should handle replication of input.
            # Therefore, we assume the (replicated) input batch to be divisible by the number of parallel channels.
            if residual_input.shape[0] % adapter_setup.parallel_channels != 0:
                raise ValueError(
                    "The total input batch size in a Parallel adapter block must be divisible by the number of"
                    " parallel channels."
                )
            orig_batch_size = residual_input.shape[0] // adapter_setup.parallel_channels

        # sequentially feed different parts of the blown-up batch into different adapters
        children_outputs = []
        # track which prefix is longest for padding in the end
        max_prefix_length = 0
        for i, child in enumerate(adapter_setup):
            # construct inputs to child modules
            inputs = {
                "key_states": key_states[i * orig_batch_size : (i + 1) * orig_batch_size],
                "value_states": value_states[i * orig_batch_size : (i + 1) * orig_batch_size],
                "residual_input": residual_input[i * orig_batch_size : (i + 1) * orig_batch_size],
                "attention_mask": attention_mask[i * orig_batch_size : (i + 1) * orig_batch_size]
                if attention_mask is not None
                else None,
                "invert_mask": invert_mask,
                "idx_range": idx_range,
            }

            # Case 1: We have a nested stack -> call stack method
            if isinstance(child, Stack):
                child_outputs = self.adapter_stack(
                    child,
                    **inputs,
                    lvl=lvl + 1,
                )
                children_outputs.append(child_outputs)
            # Case 2. We have a nested batchsplit block -> call batchsplit method
            elif isinstance(child, BatchSplit):
                child_outputs = self.adapter_batchsplit(
                    child,
                    **inputs,
                    lvl=lvl + 1,
                )
                children_outputs.append(child_outputs)
            # Case 3: We have a single adapter which is part of this module -> forward pass
            elif child in self.prefixes:
                child_outputs = self.single_forward(
                    child,
                    **inputs,
                )
                children_outputs.append(child_outputs)
            # Case 4: nesting other composition blocks is invalid
            elif isinstance(child, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        child.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # As all prefix tuning modules are centrally stored, fail if not found.
            else:
                raise ValueError(f"Unknown prefix tuning name '{child}'.")

            # update max prefix length
            current_prefix_length = child_outputs[0].shape[-2]
            if current_prefix_length > max_prefix_length:
                max_prefix_length = current_prefix_length

        # concatenate all outputs and return
        key_states, value_states, residual_input, attention_mask = self._pad_and_concat(
            max_prefix_length, children_outputs, invert_mask=invert_mask
        )
        return key_states, value_states, residual_input, attention_mask

    def adapter_batchsplit(
        self,
        adapter_setup: BatchSplit,
        key_states,
        value_states,
        residual_input,
        attention_mask=None,
        invert_mask=True,
        idx_range=None,
        lvl=0,
    ):
        if not sum(adapter_setup.batch_sizes) == key_states.shape[0]:
            raise IndexError(
                "The given batch has a size of {} which is not compatible with batch_sizes {}".format(
                    key_states.shape[0], adapter_setup.batch_sizes
                )
            )

        children_outputs = []
        # track which prefix is longest for padding in the end
        max_prefix_length = 0
        for i, adapter_block in enumerate(adapter_setup):
            # compute ids of sequences that should be passed to the ith adapter
            if idx_range is None:
                split_idx_range = range(
                    sum(adapter_setup.batch_sizes[:i]),
                    sum(adapter_setup.batch_sizes[: i + 1]),
                )
            else:
                split_idx_range = range(
                    idx_range.start + sum(adapter_setup.batch_sizes[:i]),
                    idx_range.start + sum(adapter_setup.batch_sizes[: i + 1]),
                )
            inputs = {
                "key_states": key_states[split_idx_range],
                "value_states": value_states[split_idx_range],
                "residual_input": residual_input[split_idx_range],
                "attention_mask": attention_mask[split_idx_range] if attention_mask is not None else None,
                "invert_mask": invert_mask,
                "idx_range": split_idx_range,
            }
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                child_outputs = self.adapter_stack(
                    adapter_block,
                    **inputs,
                    lvl=lvl + 1,
                )
                children_outputs.append(child_outputs)
            # Case 2: We have a nested batch split block -> call batchsplit method
            elif isinstance(adapter_block, BatchSplit):
                child_outputs = self.adapter_batchsplit(
                    adapter_block,
                    **inputs,
                    lvl=lvl + 1,
                )
                children_outputs.append(child_outputs)
            # Case 4: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.prefixes:
                child_outputs = self.single_forward(
                    adapter_block,
                    **inputs,
                )
                children_outputs.append(child_outputs)
            # Case 5: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # As all prefix tuning modules are centrally stored, fail if not found.
            else:
                raise ValueError(f"Unknown prefix tuning name '{adapter_block}'.")

            # update max prefix length
            current_prefix_length = child_outputs[0].shape[-2]
            if current_prefix_length > max_prefix_length:
                max_prefix_length = current_prefix_length

        # concatenate all outputs and return
        key_states, value_states, residual_input, attention_mask = self._pad_and_concat(
            max_prefix_length, children_outputs, invert_mask=invert_mask
        )
        return key_states, value_states, residual_input, attention_mask

    def forward(self, key_states, value_states, residual_input, attention_mask=None, invert_mask=True):
        adapter_setup = self.get_active_setup(self.prefixes)
        if adapter_setup is not None:
            if isinstance(adapter_setup, Stack):
                key_states, value_states, _, attention_mask = self.adapter_stack(
                    adapter_setup,
                    key_states,
                    value_states,
                    residual_input,
                    attention_mask=attention_mask,
                    invert_mask=invert_mask,
                )
            elif isinstance(adapter_setup, Parallel):
                key_states, value_states, _, attention_mask = self.adapter_parallel(
                    adapter_setup,
                    key_states,
                    value_states,
                    residual_input,
                    attention_mask=attention_mask,
                    invert_mask=invert_mask,
                )
            elif isinstance(adapter_setup, BatchSplit):
                key_states, value_states, _, attention_mask = self.adapter_batchsplit(
                    adapter_setup,
                    key_states,
                    value_states,
                    residual_input,
                    attention_mask=attention_mask,
                    invert_mask=invert_mask,
                )
            else:
                raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with prefix tuning.")

        return key_states, value_states, attention_mask
