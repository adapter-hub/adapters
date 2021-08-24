from abc import ABC, abstractmethod
from typing import List, Mapping, Union

import torch
from torch import nn

from .composition import AdapterCompositionBlock, BatchSplit, Fuse, Parallel, Split, Stack, parse_composition
from .modeling import Adapter, BertFusion


class AdapterLayerBaseMixin(ABC):
    """
    An abstract base implementation of adapter integration into a Transformer block. In BERT, subclasses of this module
    are placed in the BertSelfOutput module and in the BertOutput module.
    """

    # override this property if layer norm has a different name
    @property
    def transformer_layer_norm(self):
        return self.LayerNorm

    @property
    @abstractmethod
    def adapter_config_key(self):
        """Gets the name of the key by which this adapter location is identified in the adapter configuration."""
        pass

    @property
    def layer_idx(self):
        return getattr(self, "_layer_idx", -1)

    @layer_idx.setter
    def layer_idx(self, layer_idx):
        idx = getattr(self, "_layer_idx", layer_idx)
        assert idx == layer_idx
        setattr(self, "_layer_idx", idx)

    def _init_adapter_modules(self):
        self.adapters = nn.ModuleDict(dict())
        self.adapter_fusion_layer = nn.ModuleDict(dict())

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.layer_idx = layer_idx
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config.get(self.adapter_config_key, None):
            reduction_factor = adapter_config["reduction_factor"]
            if isinstance(reduction_factor, Mapping):
                if str(self.layer_idx) in reduction_factor:
                    reduction_factor = reduction_factor[str(self.layer_idx)]
                elif "default" in reduction_factor:
                    reduction_factor = reduction_factor["default"]
                else:
                    raise KeyError(
                        "The given reduction factor mapping does not give a default value and does not specify each "
                        "reduction factor individually. You need to provide a default value like this: "
                        '{"1": 16, "default": 16}'
                    )

            adapter = Adapter(
                input_size=self.config.hidden_size,
                down_sample=self.config.hidden_size // reduction_factor,
                add_layer_norm_before=adapter_config["ln_before"],
                add_layer_norm_after=adapter_config["ln_after"],
                non_linearity=adapter_config["non_linearity"],
                residual_before_ln=adapter_config["adapter_residual_before_ln"],
            )
            adapter.train(self.training)  # make sure training mode is consistent
            self.adapters[adapter_name] = adapter

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.adapters:
            del self.adapters[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        """See BertModel.add_fusion_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        if self.config.adapters.common_config_value(adapter_names, self.adapter_config_key):
            fusion_config = self.config.adapters.get_fusion(adapter_names)
            fusion = BertFusion(
                fusion_config,
                self.config.hidden_size,
                self.config.attention_probs_dropout_prob,
            )
            fusion.train(self.training)  # make sure training mode is consistent
            self.adapter_fusion_layer[",".join(adapter_names)] = fusion

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        adapter_names = adapter_names if isinstance(adapter_names, str) else ",".join(adapter_names)
        if adapter_names in self.adapter_fusion_layer:
            del self.adapter_fusion_layer[adapter_names]

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        """
        Unfreezes a given list of adapters, the adapter fusion layer, or both

        Args:
            adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
            unfreeze_adapters: whether the adapters themselves should be unfreezed
            unfreeze_fusion: whether the adapter attention layer for the given adapters should be unfreezed
        """
        if unfreeze_adapters:
            for adapter_name in adapter_setup.flatten():
                if adapter_name in self.adapters:
                    for param in self.adapters[adapter_name].parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_setup, Fuse):
                if adapter_setup.name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[adapter_setup.name].parameters():
                        param.requires_grad = True
            for sub_setup in adapter_setup:
                if isinstance(sub_setup, Fuse):
                    if sub_setup.name in self.adapter_fusion_layer:
                        for param in self.adapter_fusion_layer[sub_setup.name].parameters():
                            param.requires_grad = True

    def get_adapter_preparams(
        self,
        adapter_config,
        hidden_states,
        input_tensor,
        fusion_config=None,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuratio

        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None

        if adapter_config["residual_before_ln"]:
            residual = hidden_states

        if fusion_config is not None and fusion_config["query_before_ln"]:
            query = hidden_states

        if adapter_config["original_ln_before"]:
            if self.transformer_layer_norm:
                hidden_states = self.transformer_layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        if not adapter_config["residual_before_ln"]:
            residual = hidden_states

        if fusion_config is not None and not fusion_config["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def adapter_stack(self, adapter_setup: Stack, hidden_states, input_tensor, lvl=0):
        """
        Forwards the given input through the given stack of adapters.
        """
        for i, adapter_stack_layer in enumerate(adapter_setup):
            # Break if setup is too deep
            if isinstance(adapter_stack_layer, AdapterCompositionBlock) and lvl >= 1:
                raise ValueError(
                    "Specified adapter setup is too deep. Cannot have {} at level {}".format(
                        adapter_stack_layer.__class__.__name__, lvl
                    )
                )
            # Case 1: We have a nested fusion layer -> call fusion method
            if isinstance(adapter_stack_layer, Fuse):
                hidden_states = self.adapter_fusion(adapter_stack_layer, hidden_states, input_tensor, lvl=lvl + 1)
            # Case 2: We have a nested split layer -> call split method
            elif isinstance(adapter_stack_layer, Split):
                hidden_states = self.adapter_split(adapter_stack_layer, hidden_states, input_tensor, lvl=lvl + 1)
            # Case 3: We have a nested parallel layer -> call parallel method
            elif isinstance(adapter_stack_layer, Parallel):
                hidden_states, input_tensor = self.adapter_parallel(
                    adapter_stack_layer, hidden_states, input_tensor, lvl=lvl + 1
                )
            # Case 4: We have a nested batch split block -> call batchsplit method
            elif isinstance(adapter_stack_layer, BatchSplit):
                hidden_states = self.adapter_batchsplit(adapter_stack_layer, hidden_states, input_tensor, lvl=lvl + 1)
            # Case 5: We have a single adapter which is part of this module -> forward pass
            elif adapter_stack_layer in self.adapters:
                adapter_layer = self.adapters[adapter_stack_layer]
                adapter_config = self.config.adapters.get(adapter_stack_layer)
                hidden_states, _, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)
                hidden_states, _, up = adapter_layer(hidden_states, residual_input=residual)
                # as this stack might be part of a fusion block, return the adapter up-projection output here
                # together with the final output (with potential residuals & norms) if we reached the last block of the stack
                if i == len(adapter_setup) - 1:
                    return hidden_states, up, input_tensor
            # Case X: No adapter which is part of this module -> ignore

        # If we got here, we either had another nested composition block
        # or no adapter was found. In both cases, we don't need to set the second return value for fusion
        return hidden_states, None, input_tensor

    def adapter_fusion(self, adapter_setup: Fuse, hidden_states, input_tensor, lvl=0):
        """
        Performs adapter fusion with the given adapters for the given input.
        """
        # config of _last_ fused adapter is significant
        adapter_config = self.config.adapters.get(adapter_setup.last())
        fusion_config = self.config.adapters.get_fusion(adapter_setup.name)
        hidden_states, query, residual = self.get_adapter_preparams(
            adapter_config, hidden_states, input_tensor, fusion_config=fusion_config
        )

        up_list = []

        for adapter_block in adapter_setup:
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                _, up, _ = self.adapter_stack(adapter_block, hidden_states, input_tensor, lvl=lvl + 1)
                if up is not None:  # could be none if stack is empty
                    up_list.append(up)
            # Case 2: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapters:
                adapter_layer = self.adapters[adapter_block]
                _, _, up = adapter_layer(hidden_states, residual_input=residual)
                up_list.append(up)
            # Case 3: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore

        if len(up_list) > 0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)

            hidden_states = self.adapter_fusion_layer[adapter_setup.name](
                query,
                up_list,
                up_list,
                residual,
            )

        return hidden_states

    def adapter_split(self, adapter_setup: Split, hidden_states, input_tensor, lvl=0):
        """
        Splits the given input between the given adapters.
        """
        # config of _first_ of splitted adapters is significant
        adapter_config = self.config.adapters.get(adapter_setup.first())
        hidden_states, _, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        # split hidden representations and residuals at split index
        split_hidden_states = [
            hidden_states[:, : adapter_setup.split_index, :],
            hidden_states[:, adapter_setup.split_index :, :],
        ]
        split_input_tensor = [
            input_tensor[:, : adapter_setup.split_index, :],
            input_tensor[:, adapter_setup.split_index :, :],
        ]
        split_residual = [
            residual[:, : adapter_setup.split_index, :],
            residual[:, adapter_setup.split_index :, :],
        ]

        for i, adapter_block in enumerate(adapter_setup):
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                split_hidden_states[i], _, _ = self.adapter_stack(
                    adapter_block, split_hidden_states[i], split_input_tensor[i], lvl=lvl + 1
                )
            # Case 2: We have a nested split -> recursively call split
            elif isinstance(adapter_block, Split):
                split_hidden_states[i] = self.adapter_split(
                    adapter_block, split_hidden_states[i], split_input_tensor[i], lvl=lvl + 1
                )
            # Case 3: We have a nested batch split -> call batch split method
            elif isinstance(adapter_block, BatchSplit):
                split_hidden_states[i] = self.adapter_batchsplit(
                    adapter_block, split_hidden_states[i], split_input_tensor[i], lvl=lvl + 1
                )
            # Case 4: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapters:
                adapter_layer = self.adapters[adapter_block]
                split_hidden_states[i], _, _ = adapter_layer(split_hidden_states[i], residual_input=split_residual[i])
            # Case 5: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore

        hidden_states = torch.cat(split_hidden_states, dim=1)
        return hidden_states

    def adapter_parallel(self, adapter_setup: Parallel, hidden_states, input_tensor, lvl=0):
        """
        For parallel execution of the adapters on the same input. This means that the input is repeated N times before
        feeding it to the adapters (where N is the number of adapters).
        """
        # We assume that all adapters have the same config
        adapter_config = self.config.adapters.get(adapter_setup.first())

        if not self.config.adapters.is_parallelized:
            orig_batch_size = input_tensor.shape[0]
            input_tensor = input_tensor.repeat(self.config.adapters.active_setup.parallel_channels, 1, 1)
            hidden_states = hidden_states.repeat(self.config.adapters.active_setup.parallel_channels, 1, 1)
            self.config.adapters.is_parallelized = True
        else:
            # The base model should handle replication of input.
            # Therefore, we assume the (replicated) input batch to be divisible by the number of parallel channels.
            if hidden_states.shape[0] % adapter_setup.parallel_channels != 0:
                raise ValueError(
                    "The total input batch size in a Parallel adapter block must be divisible by the number of parallel channels."
                )
            orig_batch_size = hidden_states.shape[0] // adapter_setup.parallel_channels

        hidden_states, _, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        # sequentially feed different parts of the blown-up batch into different adapters
        children_hidden = []
        for i, child in enumerate(adapter_setup):
            # Case 1: We have a nested stack -> call stack method
            if isinstance(child, Stack):
                child_hidden_states, _, _ = self.adapter_stack(
                    child,
                    hidden_states[i * orig_batch_size : (i + 1) * orig_batch_size],
                    input_tensor[i * orig_batch_size : (i + 1) * orig_batch_size],
                    lvl=lvl + 1,
                )
                children_hidden.append(child_hidden_states)
            # Case 2. We have a nested batchsplit block -> call batchsplit method
            elif isinstance(child, BatchSplit):
                child_hidden_states = self.adapter_batchsplit(
                    child,
                    hidden_states[i * orig_batch_size : (i + 1) * orig_batch_size],
                    input_tensor[i * orig_batch_size : (i + 1) * orig_batch_size],
                    lvl=lvl + 1,
                )
                children_hidden.append(child_hidden_states)
            # Case 3: We have a single adapter which is part of this module -> forward pass
            elif child in self.adapters:
                adapter_layer = self.adapters[child]
                child_hidden_states, _, _ = adapter_layer(
                    hidden_states[i * orig_batch_size : (i + 1) * orig_batch_size],
                    residual_input=residual[i * orig_batch_size : (i + 1) * orig_batch_size],
                )
                children_hidden.append(child_hidden_states)
            # Case 4: nesting other composition blocks is invalid
            elif isinstance(child, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        child.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore
            else:
                children_hidden.append(hidden_states[i * orig_batch_size : (i + 1) * orig_batch_size])

        # concatenate all outputs and return
        hidden_states = torch.cat(children_hidden, 0)
        return hidden_states, input_tensor

    def adapter_batchsplit(self, adapter_setup: BatchSplit, hidden_states, input_tensor, lvl=0):
        if not sum(adapter_setup.batch_sizes) == hidden_states.shape[0]:
            raise IndexError(
                "The given batch has a size of {} which is not compatible with batch_sizes {}".format(
                    hidden_states.shape[0], adapter_setup.batch_sizes
                )
            )

        adapter_config = self.config.adapters.get(adapter_setup.first())
        hidden_states, _, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)
        children_hidden = []
        for i, adapter_block in enumerate(adapter_setup):
            # compute ids of sequences thet should be passed to the ith adapter
            batch_idx = (
                sum(adapter_setup.batch_sizes[:i]),
                sum(adapter_setup.batch_sizes[: i + 1]),
            )
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                child, _, _ = self.adapter_stack(
                    adapter_block,
                    hidden_states[batch_idx[0] : batch_idx[1]],
                    input_tensor[batch_idx[0] : batch_idx[1]],
                    lvl=lvl + 1,
                )
                children_hidden.append(child)
            # Case 2: We have a nested split -> recursively call split
            elif isinstance(adapter_block, Split):
                child = self.adapter_split(
                    adapter_block,
                    hidden_states[batch_idx[0] : batch_idx[1]],
                    input_tensor[batch_idx[0] : batch_idx[1]],
                    lvl=lvl + 1,
                )
                children_hidden.append(child)
            # Case 3: We have a nested batch split block -> call batchsplit method
            elif isinstance(adapter_block, BatchSplit):
                child = self.adapter_batchsplit(
                    adapter_block,
                    hidden_states[batch_idx[0] : batch_idx[1]],
                    input_tensor[batch_idx[0] : batch_idx[1]],
                    lvl=lvl + 1,
                )
                children_hidden.append(child)
            # Case 4: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapters:

                adapter_layer = self.adapters[adapter_block]
                child, _, _ = adapter_layer(
                    hidden_states[batch_idx[0] : batch_idx[1]], residual_input=residual[batch_idx[0] : batch_idx[1]]
                )
                children_hidden.append(child)
            # Case 5: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore
            else:
                children_hidden.append(hidden_states[batch_idx])

        hidden_states = torch.cat(children_hidden, 0)
        return hidden_states

    def adapters_forward(self, hidden_states, input_tensor, **kwargs):
        """
        Called for each forward pass through adapters.
        """
        if hasattr(self.config, "adapters"):
            # First check for given arguments before falling back to defined setup
            adapter_setup = kwargs.pop("adapter_names", None)
            if adapter_setup is not None:
                adapter_setup = parse_composition(adapter_setup)
            else:
                adapter_setup = self.config.adapters.active_setup
        else:
            adapter_setup = None
        skip_adapters = adapter_setup is None or (
            self.config.adapters.skip_layers is not None and self.layer_idx in self.config.adapters.skip_layers
        )
        if not skip_adapters and (len(set(self.adapters.keys()) & adapter_setup.flatten()) > 0):
            if isinstance(adapter_setup, Stack):
                hidden_states, _, input_tensor = self.adapter_stack(adapter_setup, hidden_states, input_tensor)
            elif isinstance(adapter_setup, Fuse):
                hidden_states = self.adapter_fusion(adapter_setup, hidden_states, input_tensor)
            elif isinstance(adapter_setup, Split):
                hidden_states = self.adapter_split(adapter_setup, hidden_states, input_tensor)
            elif isinstance(adapter_setup, Parallel):
                # notice that we are overriding input tensor here to keep the same dim as hidden_states for the residual
                # in case we were blowing up the batch for parallel processing of multiple adapters for the same input
                hidden_states, input_tensor = self.adapter_parallel(adapter_setup, hidden_states, input_tensor)
            elif isinstance(adapter_setup, BatchSplit):
                hidden_states = self.adapter_batchsplit(adapter_setup, hidden_states, input_tensor)
            else:
                raise ValueError(f"Invalid adapter setup {adapter_setup}")

            last_config = self.config.adapters.get(adapter_setup.last())
            if last_config["original_ln_after"]:
                if self.transformer_layer_norm:
                    hidden_states = self.transformer_layer_norm(hidden_states + input_tensor)
                else:
                    hidden_states = hidden_states + input_tensor

        elif self.transformer_layer_norm:
            hidden_states = self.transformer_layer_norm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor

        return hidden_states
