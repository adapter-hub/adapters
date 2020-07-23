import copy
import json
import logging
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, asdict, dataclass, field, is_dataclass, replace
from os.path import isfile
from typing import List, Optional, Union

from .adapter_utils import AdapterType, get_adapter_config_hash, resolve_adapter_config


logger = logging.getLogger(__name__)


@dataclass()
class InvertibleAdapterConfig(Mapping):
    block_type: str
    non_linearity: str
    reduction_factor: int

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise FrozenInstanceError()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)


@dataclass
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

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise FrozenInstanceError()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def to_dict(self):
        return asdict(self)

    def replace(self, **changes):
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, config):
        return cls(**config)

    @classmethod
    def load(cls, config: Union[dict, str], download_kwargs=None, **kwargs):
        """Loads a given adapter configuration specifier into a full AdapterConfig instance.

        Args:
            config (Union[dict, str]): The configuration to load. Can be either:
                - a dictionary representing the full config
                - an identifier string available in ADAPTER_CONFIG_MAP
                - the path to a file containing a full adapter configuration
                - an identifier string available in Adapter-Hub

        Returns:
            dict: The resolved adapter configuration dictionary.
        """
        # if force_download is set, skip the local map
        if download_kwargs and download_kwargs.get("force_download", False):
            local_map = None
        else:
            local_map = ADAPTER_CONFIG_MAP
        if download_kwargs:
            config_dict = resolve_adapter_config(config, local_map=local_map, **download_kwargs)
        else:
            config_dict = resolve_adapter_config(config, local_map=local_map)
        # convert back to dict to allow attr overrides
        if isinstance(config_dict, AdapterConfig):
            config_dict = config_dict.to_dict()
        config_dict.update(kwargs)
        return AdapterConfig.from_dict(config_dict)


@dataclass
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
    non_linearity: str = "relu"
    reduction_factor: int = 16
    invertible_adapter: Optional[dict] = InvertibleAdapterConfig(
        block_type="nice", non_linearity="relu", reduction_factor=2
    )


@dataclass
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
    non_linearity: str = "swish"
    reduction_factor: int = 16


ADAPTER_CONFIG_MAP = {"pfeiffer": PfeifferConfig(), "houlsby": HoulsbyConfig()}

DEFAULT_ADAPTER_CONFIG = "pfeiffer"


class ModelAdaptersConfig:
    """This class manages the setup and configuration of adapter modules in a pre-trained model.
    """

    def __init__(self, **kwargs):
        # adapters maps <name> -> (<type>, <config_name>)
        self.adapters = kwargs.pop("adapters", {})
        self.config_map = kwargs.pop("config_map", {})

    def adapter_list(self, adapter_type: AdapterType) -> list:
        return [k for k, v in self.adapters.items() if v[0] == adapter_type]

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
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.config_map[config_name] = config
        self.adapters[adapter_name] = (adapter_type, config_name)
        logger.info(f"Adding adapter '{adapter_name}' of type '{adapter_type}'.")

    def get_config(self, adapter_type: AdapterType) -> dict:
        config = self.config_map.get(adapter_type, None)
        if isinstance(config, str) and config in ADAPTER_CONFIG_MAP:
            return ADAPTER_CONFIG_MAP[config]
        return config

    def set_config(self, adapter_type: AdapterType, config: Union[dict, str, AdapterConfig]):
        """Sets the default adapter configuration of the specified adapter type.

        Args:
            config (str or dict or AdapterConfig): adapter configuration, can be either:
                - a string identifying a pre-defined adapter configuration
                - a dictionary representing the adapter configuration
                - the path to a file containing the adapter configuration
        """
        assert len(self.adapter_list(adapter_type)) < 1, "Can only set new config if no adapters have been added."

        if isinstance(config, Mapping) or config in ADAPTER_CONFIG_MAP:
            self.config_map[adapter_type] = config
        elif isfile(config):
            with open(config, "r", encoding="utf-8") as f:
                self.config_map[adapter_type] = json.load(f)
        else:
            raise ValueError("Unable to identify {} as a valid adapter config.".format(config))

    def common_config(self, adapter_names: list) -> Optional[AdapterConfig]:
        """Checks whether all adapters in a list share the same adapter configuration"""
        common_config = None
        for name in adapter_names:
            adapter_config = self.get(name)
            if not is_dataclass(adapter_config):
                adapter_config = AdapterConfig.from_dict(adapter_config)
            if common_config and adapter_config != adapter_config:
                return None
            common_config = adapter_config
        return common_config

    def to_dict(self):
        output_dict = {}
        output_dict["adapters"] = copy.deepcopy(self.adapters)
        return output_dict


def build_full_config(adapter_config, model_config, **kwargs):
    config_dict = {"model_type": model_config.model_type, "hidden_size": model_config.hidden_size}
    config_dict.update(kwargs)
    if is_dataclass(adapter_config):
        config_dict["config"] = adapter_config.to_dict()
    else:
        config_dict["config"] = adapter_config
    return config_dict


@dataclass
class AdapterFusionConfig(Mapping):
    """Base class that models the architecture of an adapter."""

    key: bool
    query: bool
    value: bool
    query_before_ln: bool
    regularization: bool
    residual_before: bool
    temperature: bool
    value_before_softmax: bool
    value_initialized: str

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise FrozenInstanceError()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def to_dict(self):
        return asdict(self)

    def replace(self, **changes):
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, config):
        return cls(**config)

    @classmethod
    def load(cls, config: Union[dict, str], **kwargs):
        """Loads a given adapter configuration specifier into a full AdapterConfig instance.

        Args:
            config (Union[dict, str]): The configuration to load. Can be either:
                - a dictionary representing the full config
                - an identifier string available in ADAPTER_CONFIG_MAP
                - the path to a file containing a full adapter configuration
                - an identifier string available in Adapter-Hub

        Returns:
            dict: The resolved adapter configuration dictionary.
        """
        # currently storing AdapterFusion weights on AdapterHub is not supported.
        config_dict = resolve_adapter_config(config, local_map=ADAPTERFUSION_CONFIG_MAP, try_loading_from_hub=False)
        # convert back to dict to allow attr overrides
        if isinstance(config_dict, AdapterFusionConfig):
            config_dict = config_dict.to_dict()
        config_dict.update(kwargs)
        return AdapterFusionConfig.from_dict(config_dict)


@dataclass
class StaticAdapterFusionConfig(AdapterFusionConfig):
    """
    The adapter architecture proposed by Houlsby et. al., 2019.
    Described in https://arxiv.org/pdf/1902.00751.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = False
    query_before_ln: bool = False
    regularization: bool = False
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    value_initialized: str = False


@dataclass
class DynamicAdapterFusionConfig(AdapterFusionConfig):
    """
    The adapter architecture proposed by Houlsby et. al., 2019.
    Described in https://arxiv.org/pdf/1902.00751.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    value_initialized: str = True


ADAPTERFUSION_CONFIG_MAP = {"static": StaticAdapterFusionConfig(), "dynamic": DynamicAdapterFusionConfig()}

DEFAULT_ADAPTERFUSION_CONFIG = "dynamic"
