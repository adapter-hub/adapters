from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

import numpy as np
from torch import nn

from ..composition import AdapterCompositionBlock
from ..context import AdapterSetup, ForwardContext


# We don't inherit from ABC because __slots__ changes object layout
class AdapterLayerBase(metaclass=ABCMeta):
    """
    Base class for all adaptation methods that require per-layer modules.
    """

    @property
    def layer_idx(self):
        return getattr(self, "_layer_idx", -1)

    @layer_idx.setter
    def layer_idx(self, layer_idx):
        idx = getattr(self, "_layer_idx", layer_idx)
        assert idx == layer_idx
        setattr(self, "_layer_idx", idx)

    def get_active_setup(self, module_dict):
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
        if not skip_adapters and (len(set(module_dict.keys()) & adapter_setup.flatten()) > 0):
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
        raise NotImplementedError()

    @abstractmethod
    def average_adapter(self, adapter_name: str, input_adapters: Dict[str, float]) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def delete_adapter(self, adapter_name: str):
        raise NotImplementedError()

    @abstractmethod
    def add_fusion_layer(self, adapter_names: Union[List, str]):
        raise NotImplementedError()

    @abstractmethod
    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        raise NotImplementedError()

    @abstractmethod
    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        raise NotImplementedError()

    @abstractmethod
    def get_adapter(self, adapter_name: str) -> nn.Module:
        raise NotImplementedError()
