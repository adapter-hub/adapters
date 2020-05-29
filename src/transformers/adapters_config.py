from enum import Enum
import json
from os.path import isfile
import copy
from typing import Optional, Union


# TODO add more default configs here
ADAPTER_CONFIG_MAP = {
    'pfeiffer': {
        'LN_after': False,
        'LN_before': False,
        'MH_Adapter': False,
        'Output_Adapter': True,
        'adapter_residual_before_ln': False,
        'attention_type': 'sent-lvl-dynamic',
        'new_attention_norm': False,
        'non_linearity': 'relu',
        'original_ln_after': True,
        'original_ln_before': True,
        'reduction_factor': 16,
        'residual_before_ln': True,
        'invertible_adapter': {
            'block_type': 'nice',
            'non_linearity': 'relu',
            'reduction_factor': 2
        }
    },
    'houlsby': {
        "LN_after": False,
        "LN_before": False,
        "MH_Adapter": True,
        "Output_Adapter": True,
        "adapter_residual_before_ln": False,
        "attention_type": "sent-lvl-dynamic",
        "new_attention_norm": False,
        "non_linearity": "swish",
        "original_ln_after": True,
        "original_ln_before": False,
        "reduction_factor": 16,
        "residual_before_ln": True,
    }
}

DEFAULT_ADAPTER_CONFIG = 'pfeiffer'


class AdapterType(str, Enum):
    """Models all currently available model adapter types."""

    text_task = "text_task"
    text_lang = "text_lang"

    @classmethod
    def has(cls, value):
        return value in cls.__members__.values()

    def __repr__(self):
        return self.value


class ModelAdaptersConfig:
    def __init__(self, **kwargs):
        self.adapters = kwargs.pop("adapters", {})
        self.config_map = kwargs.pop("config_map", {})

    def adapter_list(self, adapter_type: AdapterType) -> list:
        return [
            k for k, v in self.adapters.items() if v['type'] == adapter_type
        ]

    def get_type(self, adapter_name: str) -> Optional[AdapterType]:
        if adapter_name in self.adapters:
            return self.adapters[adapter_name]['type']
        else:
            return None

    def get(self, adapter_name: str, return_type: bool = False):
        if adapter_name in self.adapters:
            adapter = self.adapters[adapter_name]
            config = adapter['config']
            adapter_type = adapter['type']
            if not config:
                config = self.config_map[adapter['type']]
            if isinstance(config, str):
                config = ADAPTER_CONFIG_MAP[config]
        else:
            config, adapter_type = None, None
        if return_type:
            return config, adapter_type
        else:
            return config

    def add(self, adapter_name: str, adapter_type: AdapterType, config=None):
        if adapter_name in self.adapters:
            raise ValueError(f"An adapter with the name '{adapter_name}' has already been added.")
        # TODO temporary, remove when multiple adapter configs are supported (!)
        assert config is None, "All adapters of one type must have the same config."
        self.adapters[adapter_name] = {
            'type': adapter_type,
            'config': config
        }

    def get_config(self, adapter_type: AdapterType) -> dict:
        config = self.config_map.get(adapter_type, None)
        if isinstance(config, str) and config in ADAPTER_CONFIG_MAP:
            return ADAPTER_CONFIG_MAP[config]
        return config

    def set_config(self, adapter_type: AdapterType, config: Union[dict, str]):
        """Sets the default adapter configuration of the specified adapter type.

        Args:
            config (str or dict): adapter configuration, can be either:
                - a string identifying a pre-defined adapter configuration
                - a dictionary representing the adapter configuration
                - the path to a file containing the adapter configuration
        """
        assert len(self.adapter_list(adapter_type)) < 1, "Can only set new config if no adapters have been added."
        if isinstance(config, dict) or config in ADAPTER_CONFIG_MAP:
            self.config_map[adapter_type] = config
        elif isfile(config):
            with open(config, 'r', encoding='utf-8') as f:
                self.config_map[adapter_type] = json.load(f)
        else:
            raise ValueError("Unable to identify {} as a valid adapter config.".format(config))

    def to_dict(self):
        output_dict = {}
        output_dict['adapters'] = copy.deepcopy(self.adapters)
        output_dict['config_map'] = copy.deepcopy(self.config_map)
        return output_dict


def build_full_config(adapter_config, adapter_type, model_config, name=None, with_head=False):
    config_dict = {
        'type': adapter_type,
        'model_type': model_config.model_type,
        'hidden_size': model_config.hidden_size
    }
    if name:
        config_dict['name'] = name
    config_dict['config'] = adapter_config
    if with_head:
        config_dict['prediction_head'] = model_config.prediction_heads[name]
    return config_dict
