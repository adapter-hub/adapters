from collections.abc import Mapping
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
import json
import logging
from os.path import isfile
import copy
from typing import Optional, Union, List
import hashlib


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InvertibleAdapterConfig(Mapping):
    block_type: str
    non_linearity: str
    reduction_factor: int

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)


@dataclass(frozen=True)
class AdapterConfig(Mapping):
    """Base class that models the architecture of an adapter."""
    original_ln_before: bool
    original_ln_after: bool
    residual_before_ln: bool
    adapter_residual_before_ln: bool
    ln_before: bool
    ln_after: bool
    mh_adapter: bool
    output_adapter: bool
    non_linearity: str
    reduction_factor: int
    invertible_adapter: Optional[InvertibleAdapterConfig] = None
    leave_out: List[int] = field(default_factory=list)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config):
        # remove all invalid keys
        valid_dict = {}
        for k, v in config.items():
            if k in cls.__annotations__:
                valid_dict[k] = v
        return cls(**valid_dict)


@dataclass(frozen=True)
class PfeifferConfig(AdapterConfig):
    """
    The adapter architecture proposed by Pfeiffer et. al., 2020.
    Described in https://arxiv.org/pdf/2005.00247.pdf.
    """
    original_ln_before: bool = True
    original_ln_after: bool = True
    residual_before_ln: bool = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = 'relu'
    reduction_factor: int = 16
    invertible_adapter: Optional[dict] = InvertibleAdapterConfig(
        block_type='nice',
        non_linearity='relu',
        reduction_factor=2
    )


@dataclass(frozen=True)
class HoulsbyConfig(AdapterConfig):
    """
    The adapter architecture proposed by Houlsby et. al., 2019.
    Described in https://arxiv.org/pdf/1902.00751.pdf.
    """
    original_ln_before: bool = False
    original_ln_after: bool = True
    residual_before_ln: bool = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = True
    output_adapter: bool = True
    non_linearity: str = 'swish'
    reduction_factor: int = 16


ADAPTER_CONFIG_MAP = {
    'pfeiffer': PfeifferConfig(),
    'houlsby': HoulsbyConfig()
}

DEFAULT_ADAPTER_CONFIG = 'pfeiffer'

# these keys are ignored when calculating the config hash
ADAPTER_CONFIG_HASH_IGNORE = []


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
    """This class manages the setup and configuration of adapter modules in a pre-trained model.
    """
    def __init__(self, **kwargs):
        # adapters maps <name> -> (<type>, <config_name>)
        self.adapters = kwargs.pop("adapters", {})
        self.config_map = kwargs.pop("config_map", {})

    def adapter_list(self, adapter_type: AdapterType) -> list:
        return [
            k for k, v in self.adapters.items() if v[0] == adapter_type
        ]

    def get_type(self, adapter_name: str) -> Optional[AdapterType]:
        if adapter_name in self.adapters:
            return self.adapters[adapter_name][0]
        else:
            return None

    def get(self, adapter_name: str, return_type: bool = False):
        if adapter_name in self.adapters:
            adapter_type, config_name = self.adapters[adapter_name]
            config = self.config_map.get(config_name, None)
            if not config:
                config = self.config_map[adapter_type]
            if isinstance(config, str):
                config = ADAPTER_CONFIG_MAP[config]
        else:
            config, adapter_type = None, None
        if return_type:
            return config, adapter_type
        else:
            return config

    def add(self, adapter_name: str, adapter_type: AdapterType, config: Optional[Union[str, dict]] = None):
        if adapter_name in self.adapters:
            raise ValueError(f"An adapter with the name '{adapter_name}' has already been added.")
        config_name = config
        if isinstance(config, str):
            if config not in ADAPTER_CONFIG_MAP and config not in self.config_map:
                raise ValueError(f"Invalid adapter config identifier '{config}''")
        # if it's a dict, compute it's hash and add a new entry to the config map
        elif isinstance(config, dict):
            config_name = get_adapter_config_hash(config)
            self.config_map[config_name] = config
        self.adapters[adapter_name] = (adapter_type, config_name)
        logger.info(f"Adding adapter '{adapter_name}' of type '{adapter_type}'.")

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

    def common_config(self, adapter_names: list) -> Optional[dict]:
        common_config_name = None
        for name in adapter_names:
            _, config_name = self.adapters[name]
            if common_config_name and common_config_name != config_name:
                return None
            common_config_name = config_name
        if not common_config_name:
            return None
        config = self.config_map[common_config_name]
        if isinstance(config, str):
            return ADAPTER_CONFIG_MAP[config]
        return config

    def to_dict(self):
        output_dict = {}
        output_dict['adapters'] = copy.deepcopy(self.adapters)
        output_dict['config_map'] = copy.deepcopy(self.config_map)
        return output_dict


def _minimize_dict(d):
    if isinstance(d, Mapping):
        return {k: _minimize_dict(v) for (k, v) in d.items() if v}
    else:
        return d


def get_adapter_config_hash(config, length=16):
    """Calculates the hash of a given adapter configuration which is used to identify this configuration.

    Returns:
        str: The resulting hash of the given config dict.
    """
    minimized_config = _minimize_dict(
        {k: v for (k, v) in config.items() if k not in ADAPTER_CONFIG_HASH_IGNORE}
    )
    dict_str = json.dumps(minimized_config, sort_keys=True)
    h = hashlib.sha1()
    h.update(dict_str.encode(encoding='utf-8'))
    return h.hexdigest()[:length]


def build_full_config(adapter_config, adapter_type, model_config, model_name=None, name=None, with_head=False):
    config_dict = {
        'type': adapter_type,
        'model_type': model_config.model_type,
        'hidden_size': model_config.hidden_size
    }
    if model_name:
        config_dict['model_name'] = model_name
    if name:
        config_dict['name'] = name
    if is_dataclass(adapter_config):
        config_dict['config'] = adapter_config.to_dict()
    else:
        config_dict['config'] = adapter_config
    if with_head:
        config_dict['prediction_head'] = model_config.prediction_heads[name]
    return config_dict
