from abc import ABCMeta, abstractmethod
from typing import Collection, Dict, List, NamedTuple, Union

import numpy as np
import torch
from torch import nn

from ..composition import ALLOWED_NESTINGS, AdapterCompositionBlock, Average, BatchSplit, Fuse, Parallel, Split, Stack
from ..context import AdapterSetup, ForwardContext


# We don't inherit from ABC because __slots__ changes object layout
class AdapterLayerBase(metaclass=ABCMeta):
    """
    Base class for all adaptation methods that require per-layer modules.

    Make sure the 'adapter_modules_name' attribute is overriden in derived classes.
    """

    adapter_modules_name = ""

    @property
    def adapter_modules(self) -> Collection:
        return getattr(self, self.adapter_modules_name)

    @property
    def layer_idx(self):
        return getattr(self, "_layer_idx", -1)

    @layer_idx.setter
    def layer_idx(self, layer_idx):
        idx = getattr(self, "_layer_idx", layer_idx)
        assert idx == layer_idx
        setattr(self, "_layer_idx", idx)

    def get_active_setup(self):
        if hasattr(self, "adapters_config"):
            # First check current context before falling back to defined setup
            context = AdapterSetup.get_context()
            if context is not None:
                adapter_setup = context.adapter_setup
            else:
                adapter_setup = self.adapters_config.active_setup
        else:
            adapter_setup = None
        skip_adapters = adapter_setup is None or (
            self.adapters_config.skip_layers is not None and self.layer_idx in self.adapters_config.skip_layers
        )
        if not skip_adapters and (len(set(self.adapter_modules.keys()) & adapter_setup.flatten()) > 0):
            return adapter_setup
        else:
            return None

    def _store_gating_score(self, adapter_name, gating_score):
        context = ForwardContext.get_context()
        if context.output_adapter_gating_scores:
            gating_cache = context.adapter_gating_scores
            if self.layer_idx not in gating_cache[adapter_name]:
                gating_cache[adapter_name][self.layer_idx] = {}
            gating_score = gating_score.detach().squeeze().cpu().numpy()
            if len(gating_score.shape) == 0:
                gating_score = np.expand_dims(gating_score, axis=0)
            cache_score = gating_cache[adapter_name][self.layer_idx].get(self.location_key, None)
            if cache_score is not None:
                gating_cache[adapter_name][self.layer_idx][self.location_key] = np.column_stack(
                    (cache_score, gating_score)
                )
            else:
                gating_cache[adapter_name][self.layer_idx][self.location_key] = gating_score

    def _store_fusion_attentions(self, fusion_name, attentions):
        context = ForwardContext.get_context()
        if context.output_adapter_fusion_attentions:
            attention_cache = context.adapter_fusion_attentions
            if self.layer_idx not in attention_cache[fusion_name]:
                attention_cache[fusion_name][self.layer_idx] = {}
            attention_cache[fusion_name][self.layer_idx][self.location_key] = attentions

    @abstractmethod
    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        """Adds a new adapter module to the layer.

        Args:
            adapter_name (str): The name of the new adapter to add.
            layer_idx (int):
                The index of the adapters layer (this should be set once by the first added adapter and the kept fix).

        Returns:
            bool: True if the adapter was added, False otherwise.
        """
        raise NotImplementedError()

    def average_adapter(self, adapter_name: str, input_adapters: Dict[str, float], combine_strategy, **kwargs) -> bool:
        """Averages a set of adapter modules into a new adapter module.

        Args:
            adapter_name (str): The name of the new (averaged) adapter module to add.
            input_adapters (Dict[str, float]): Dictionary of adapter names and their corresponding weights.
            combine_strategy (str): The strategy to combine the adapters. Available strategies depend on the used adapter method, see: https://docs.adapterhub.ml/adapter_composition.html#merging-adapters
            **kwargs: Additional arguments that are specific to the combine_strategy. E.g. svd_rank for LoRA.

        Returns:
            bool: True if the adapter was added, False otherwise.
        """
        # add new adapter
        if self.add_adapter(adapter_name, self.layer_idx):
            if combine_strategy != "linear":
                # You get the adapter type from the input adapters
                raise ValueError(f"Combine strategy {combine_strategy} not supported for the chosen adapter methods.")

            # average weights linearly
            avg_state_dict = {}
            for name, weight in input_adapters.items():
                if name in self.adapter_modules:
                    module = self.adapter_modules[name]
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v
                else:
                    self.delete_adapter(adapter_name)  # clean up before raising error
                    raise ValueError("Adapter {} not found.".format(name))

            # load averaged weights
            self.adapter_modules[adapter_name].load_state_dict(avg_state_dict)

            return True

        return False

    def delete_adapter(self, adapter_name: str):
        """Deletes an adapter module from the layer.

        Args:
            adapter_name (str): The name of the adapter to delete.
        """
        if adapter_name in self.adapter_modules:
            del self.adapter_modules[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # default implementation does nothing as fusion is not applicable to all methods

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # default implementation does nothing as fusion is not applicable to all methods

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        """Enables/ disables a set of adapter modules within the layer.

        Args:
            adapter_setup (AdapterCompositionBlock): The adapter setup to enable/ disable.
            unfreeze_adapters (bool): Whether to unfreeze the adapters.
            unfreeze_fusion (bool): Whether to unfreeze the fusion layers.
        """
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.adapter_modules:
                    for param in self.adapter_modules[name].parameters():
                        param.requires_grad = True

    def freeze_adapter(self, adapter_name: str, freeze: bool = True):
        """Freezes/ unfreezes an adapter module.

        Args:
            adapter_name (str): The name of the adapter to freeze/ unfreeze.
            freeze (bool, optional): Whether to freeze the adapter. Defaults to True.
        """
        if adapter_name in self.adapter_modules:
            self.adapter_modules[adapter_name].train(not freeze)
            for param in self.adapter_modules[adapter_name].parameters():
                param.requires_grad = not freeze

    def get_adapter(self, adapter_name: str) -> nn.Module:
        """Returns the adapter module with the given name.

        Args:
            adapter_name (str): The name of the adapter module.
        """
        if adapter_name in self.adapter_modules:
            return self.adapter_modules[adapter_name]
        else:
            return None

    def pre_save_adapters(self):
        """Called before saving the adapters to disk."""
        pass


class ComposableAdapterLayerBase(AdapterLayerBase):
    """
    Base class for all adapter methods that support composition.

    Make sure the 'adapter_modules_name' and 'supported_compositions' attributes as well as all abstract methods are
    overriden in derived classes. 'allow_multi_parallelize' can be set to True to allow inputs to be parallelized
    independently multiple times. This is useful when there are multiple parallel input flows through an adapter layer
    (e.g. in LoRA).
    """

    supported_compositions = []
    allow_multi_parallelize = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mapping()

    def _init_mapping(self):
        # Mapping between composition block types and names of composition functions
        self.composition_to_func_map = {
            Stack: "compose_stack",
            Fuse: "compose_fuse",
            Split: "compose_split",
            BatchSplit: "compose_batch_split",
            Parallel: "compose_parallel",
            Average: "compose_average",
        }

    def _get_compose_func(self, composition_type: type) -> callable:
        """Retrieves the correct composition function based on the mapping in 'composition_to_func_map'."""
        return getattr(self, self.composition_to_func_map[composition_type])

    # START CUSTOMIZABLE METHODS #
    # The following methods should be implemented in derived classes.

    def _bsz(self, state: NamedTuple) -> int:
        """
        Returns the batch size of the given state.
        """
        return state[0].shape[0]

    def pre_block(self, adapter_setup: Union[AdapterCompositionBlock, str], state: NamedTuple) -> NamedTuple:
        """
        Optional state pre-processing method which is invoked before passing the state to the first child block of a
        composition. By default, this method does not contain any logic. E.g. used for bottleneck adapters to implement
        residuals and LNs.

        Args:
            adapter_setup (Union[AdapterCompositionBlock, str]): The current composition or single adapter.
            state (NamedTuple): The current state.

        Returns:
            NamedTuple: The pre-processed state.
        """
        return state

    @abstractmethod
    def vslice(self, state: NamedTuple, slice_obj: slice) -> NamedTuple:
        """Slices the given state along the batch size (vertical) dimension.
        This is e.g. used by the BatchSplit and Parallel composition blocks. IMPORTANT: Has to be implemented by all
        derived classes.

        Args:
            state (NamedTuple): The state to be sliced.
            slice_obj (slice): The slice object.

        Returns:
            NamedTuple: The sliced state.
        """
        raise NotImplementedError()

    @abstractmethod
    def pad_and_concat(self, states: List[NamedTuple]) -> NamedTuple:
        """Concatenates the given states along the batch size dimension.
        Pads the states before concatenation if necessary. This is e.g. used by the BatchSplit and Parallel composition
        blocks. IMPORTANT: Has to be implemented by all derived classes.

        Args:
            states (List[NamedTuple]): The states to be concatenated.

        Returns:
            NamedTuple: The concatenated state.
        """
        raise NotImplementedError()

    @abstractmethod
    def repeat(self, state: NamedTuple, channels: int) -> NamedTuple:
        """Repeats the given state along the batch size dimension for the given number of times.
        This is e.g. used by the Parallel composition block. IMPORTANT: Has to be implemented by all derived classes.

        Args:
            state (NamedTuple): The state to be repeated.
            channels (int): The number of times the state should be repeated.

        Returns:
            NamedTuple: The repeated state.
        """
        raise NotImplementedError()

    @abstractmethod
    def mean(self, states: List[NamedTuple], weights: torch.Tensor) -> NamedTuple:
        """Averages the given states along the batch size dimension by the given weights.
        This is e.g. used by the Average composition block. IMPORTANT: Has to be implemented by all derived classes.

        Args:
            states (List[NamedTuple]): The states to be averaged.
            weights (torch.Tensor): The averaging weights.

        Returns:
            NamedTuple: The averaged state.
        """
        raise NotImplementedError()

    @abstractmethod
    def compose_single(self, adapter_setup: str, state: NamedTuple, lvl: int = 0) -> NamedTuple:
        """Forwards the given state through the given single adapter.

        Args:
            adapter_setup (str): The name of the adapter.
            state (NamedTuple): The state to be forwarded.
            lvl (int, optional): The composition depth. Defaults to 0.

        Returns:
            NamedTuple: The state after forwarding through the adapter.
        """
        raise NotImplementedError()

    # END CUSTOMIZABLE METHODS #

    def check_composition_valid(self, parent: AdapterCompositionBlock, child: AdapterCompositionBlock, lvl: int):
        """Checks whether the given composition is valid.

        Args:
            parent (AdapterCompositionBlock): The parent composition block.
            child (AdapterCompositionBlock): The child composition block.
            lvl (int): The composition depth.

        Raises:
            ValueError: If the composition is invalid.
        """
        # Break if setup is too deep
        if isinstance(parent, Stack) and lvl >= 1:
            raise ValueError(
                "Specified adapter setup is too deep. Cannot have {} at level {}".format(child.__class__.__name__, lvl)
            )
        elif type(child) not in ALLOWED_NESTINGS[type(parent)]:
            raise ValueError(
                "Cannot nest {} inside {}. Only the following nestings are allowed: {}".format(
                    child.__class__.__name__,
                    parent.__class__.__name__,
                    ", ".join([t.__name__ for t in ALLOWED_NESTINGS[type(parent)]]),
                )
            )

    def compose_stack(self, adapter_setup: Stack, state: NamedTuple, lvl: int = 0) -> NamedTuple:
        """
        For sequentially stacking multiple adapters.
        """
        for i, adapter_stack_layer in enumerate(adapter_setup):
            if isinstance(adapter_stack_layer, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, adapter_stack_layer, lvl)
                composition_func = self._get_compose_func(type(adapter_stack_layer))
                state = composition_func(adapter_stack_layer, state, lvl=lvl + 1)
            elif adapter_stack_layer in self.adapter_modules:
                state = self.pre_block(adapter_stack_layer, state)
                state = self.compose_single(adapter_stack_layer, state, lvl=lvl + 1)
            else:
                pass

        return state

    def compose_fuse(self, adapter_setup: Fuse, state: NamedTuple, lvl: int = 0):
        """
        For fusing multiple adapters using adapter fusion. NOTE: This method has no default implementation.
        """
        # Fuse is currently only applicable to bottleneck adapters, thus don't provide a default implementation
        # If the adapter setup does not contain any of the adapter modules, return without doing anything
        if set(self.adapter_modules.keys()).isdisjoint(adapter_setup.flatten()):
            return state
        raise NotImplementedError()

    def compose_split(self, adapter_setup: Split, state: NamedTuple, lvl: int = 0):
        """
        For splitting to multiple adapters along the sequence length dimension. NOTE: This method has no default
        implementation.
        """
        # Split is currently only applicable to bottleneck adapters, thus don't provide a default implementation
        # If the adapter setup does not contain any of the adapter modules, return without doing anything
        if set(self.adapter_modules.keys()).isdisjoint(adapter_setup.flatten()):
            return state
        raise NotImplementedError()

    def compose_batch_split(self, adapter_setup: BatchSplit, state: NamedTuple, lvl: int = 0):
        """
        For splitting to multiple adapters along the batch size dimension.
        """
        if sum(adapter_setup.batch_sizes) != self._bsz(state):
            raise IndexError(
                "The given batch has a size of {} which is not equal to the sum of batch_sizes {}".format(
                    self._bsz(state), adapter_setup.batch_sizes
                )
            )

        state = self.pre_block(adapter_setup, state)

        # sequentially feed different parts of the blown-up batch into different adapters
        children_states = []
        for i, child in enumerate(adapter_setup):
            # compute ids of sequences that should be passed to the ith adapter
            batch_idx = (
                sum(adapter_setup.batch_sizes[:i]),
                sum(adapter_setup.batch_sizes[: i + 1]),
            )
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(
                    child,
                    self.vslice(state, slice(*batch_idx)),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(
                    child,
                    self.vslice(state, slice(*batch_idx)),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            else:
                children_states.append(self.vslice(state, slice(*batch_idx)))

        # concatenate all outputs and return
        state = self.pad_and_concat(children_states)
        return state

    def compose_parallel(self, adapter_setup: Parallel, state: NamedTuple, lvl: int = 0):
        """
        For parallel execution of the adapters on the same input. This means that the input is repeated N times before
        feeding it to the adapters (where N is the number of adapters).
        """

        context = ForwardContext.get_context()
        if not context.adapters_parallelized:
            orig_batch_size = self._bsz(state)
            state = self.repeat(state, adapter_setup.parallel_channels)
            context.adapters_parallelized = True
            context.original_batch_size = orig_batch_size
        else:
            bsz = self._bsz(state)
            # If the input was already parallelized, we can parallelize it again.
            # This is useful e.g. for LoRA, where attention matrices are parallelized independently.
            if self.allow_multi_parallelize and bsz == getattr(context, "original_batch_size", -1):
                state = self.repeat(state, adapter_setup.parallel_channels)
                orig_batch_size = bsz
            # The base model should handle replication of input.
            # Therefore, we assume the (replicated) input batch to be divisible by the number of parallel channels.
            elif bsz % adapter_setup.parallel_channels != 0:
                raise ValueError(
                    "The total input batch size in a Parallel adapter block must be divisible by the number of"
                    " parallel channels."
                )
            else:
                orig_batch_size = bsz // adapter_setup.parallel_channels

        state = self.pre_block(adapter_setup, state)

        # sequentially feed different parts of the blown-up batch into different adapters
        children_states = []
        for i, child in enumerate(adapter_setup):
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(
                    child,
                    self.vslice(state, slice(i * orig_batch_size, (i + 1) * orig_batch_size)),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(
                    child,
                    self.vslice(state, slice(i * orig_batch_size, (i + 1) * orig_batch_size)),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            else:
                children_states.append(self.vslice(state, slice(i * orig_batch_size, (i + 1) * orig_batch_size)))

        # concatenate all outputs and return
        state = self.pad_and_concat(children_states)
        return state

    def compose_average(self, adapter_setup: Average, state: NamedTuple, lvl: int = 0):
        """
        For averaging the output representations of multiple adapters.
        """

        state = self.pre_block(adapter_setup, state)

        children_states = []
        for i, child in enumerate(adapter_setup):
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            else:
                pass

        weights = torch.tensor(adapter_setup.weights)[:, None, None, None].to(state[0].device)
        state = self.mean(children_states, weights)

        return state

    def compose(self, adapter_setup: Union[AdapterCompositionBlock, str], state: NamedTuple) -> NamedTuple:
        """The main composition forward method which recursively calls the composition blocks forward methods.
        This method should be called by the forward method of the derived class.

        Args:
            adapter_setup (Union[AdapterCompositionBlock, str]): The adapter setup to be used.
            state (NamedTuple): The current state.

        Returns:
            NamedTuple: The state after forwarding through the adapter setup.
        """
        if isinstance(adapter_setup, AdapterCompositionBlock):
            composition_func = self._get_compose_func(type(adapter_setup))
            state = composition_func(adapter_setup, state, lvl=0)
        elif adapter_setup in self.adapter_modules:
            state = self.compose_single(adapter_setup, state, lvl=0)
        else:
            raise ValueError(
                "Invalid adapter setup: {} is not a valid adapter name or composition block.".format(
                    adapter_setup.__class__.__name__
                )
            )

        return state
