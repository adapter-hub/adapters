import copy
import logging
from collections.abc import Collection, Mapping
from dataclasses import FrozenInstanceError, asdict, dataclass, field, replace
from typing import List, Optional, Union

from .composition import AdapterCompositionBlock
from .utils import get_adapter_config_hash, resolve_adapter_config


logger = logging.getLogger(__name__)


class AdapterConfigBase(Mapping):
    """
    Base class for all adaptation methods. This class does not define specific configuration keys, but only provides
    some common helper methods.

    Attributes:
        architecture (str, optional): The type of adaptation method defined by the configuration.
    """

    architecture: Optional[str] = None

    def __init__(self):
        raise TypeError("AdapterConfigBase is an abstract class and cannot be instantiated.")

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

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        return asdict(self)

    def replace(self, **changes):
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, config):
        if isinstance(config, AdapterConfigBase):
            return config

        # the constructor does not accept additional kwargs, so add them separately
        defined_kwargs, new_kwargs = {}, {}
        for k, v in config.items():
            if k in cls.__dataclass_fields__.keys():
                defined_kwargs[k] = v
            else:
                new_kwargs[k] = v
        obj = cls(**defined_kwargs)
        for k, v in new_kwargs.items():
            setattr(obj, k, v)
        return obj

    @staticmethod
    def _get_config_class(config_dict):
        """
        Returns the matching config class for the given config dict based on its "architecture" key.
        """
        architecture = config_dict.get("architecture", None)
        if architecture == "prefix_tuning":
            cls_new = PrefixTuningConfig
        elif architecture == "union":
            cls_new = ConfigUnion
        else:
            cls_new = AdapterConfig

        return cls_new

    @classmethod
    def load(cls, config: Union[dict, str], download_kwargs=None, **kwargs):
        """
        Loads a given adapter configuration specifier into a full AdapterConfigBase instance.

        Args:
            config (Union[dict, str]): The configuration to load. Can be either:

                - a dictionary representing the full config
                - an identifier string available in ADAPTER_CONFIG_MAP
                - the path to a file containing a full adapter configuration
                - an identifier string available in Adapter-Hub

        Returns:
            dict: The resolved adapter configuration dictionary.
        """
        if not config:
            return None
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
        if isinstance(config_dict, AdapterConfigBase):
            cls_new = config_dict.__class__
            config_dict = config_dict.to_dict()
        else:
            cls_new = cls._get_config_class(config_dict)
        # The check for "None" is necessary because of the example script flags.
        config_dict.update((k, v) for k, v in kwargs.items() if v is not None)
        return cls_new.from_dict(config_dict)


@dataclass(eq=False)
class AdapterConfig(AdapterConfigBase):
    """
    Base class that models the architecture of an adapter.

    Args:
            reduction_factor (:obj:`int` or :obj:`Mapping`): Either an integer specifying the reduction factor for all layers
                or a mapping specifying the reduction_factor for individual layers. If not all layers are represented
                in the mapping a default value should be given e.g. {'1': 8, '6': 32, 'default': 16}
    """

    # Required options
    original_ln_before: bool
    original_ln_after: bool
    ln_before: bool
    ln_after: bool
    mh_adapter: bool
    output_adapter: bool
    non_linearity: str
    reduction_factor: Union[int, Mapping]

    # Options with defaults
    init_weights: str = "bert"
    is_parallel: bool = False
    scaling: Union[float, str] = 1.0
    residual_before_ln: bool = True
    adapter_residual_before_ln: bool = False
    inv_adapter: Optional[str] = None
    inv_adapter_reduction_factor: Optional[int] = None
    cross_adapter: bool = False
    leave_out: List[int] = field(default_factory=list)

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        elif name == "invertible_adapter":
            # This is for backwards compatibility. In v1, invertible adapters were specified in a nested config dict.
            # Now, we have two config keys directly in the adapter config.
            if value:
                object.__setattr__(self, "inv_adapter", value["block_type"])
                object.__setattr__(self, "inv_adapter_reduction_factor", value["reduction_factor"])
        else:
            object.__setattr__(self, name, value)


@dataclass(eq=False)
class PfeifferConfig(AdapterConfig):
    """
    The adapter architecture proposed by Pfeiffer et al. (2020). See https://arxiv.org/pdf/2005.00247.pdf.
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
    reduction_factor: Union[int, Mapping] = 16


@dataclass(eq=False)
class PfeifferInvConfig(PfeifferConfig):
    """
    The adapter architecture proposed by Pfeiffer et al. (2020). See https://arxiv.org/pdf/2005.00247.pdf.
    """

    inv_adapter: Optional[str] = "nice"
    inv_adapter_reduction_factor: Optional[int] = 2


@dataclass(eq=False)
class HoulsbyConfig(AdapterConfig):
    """
    The adapter architecture proposed by Houlsby et al. (2019). See https://arxiv.org/pdf/1902.00751.pdf.
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
    reduction_factor: Union[int, Mapping] = 16


@dataclass(eq=False)
class HoulsbyInvConfig(HoulsbyConfig):
    """
    The adapter architecture proposed by Houlsby et. al. (2019). See https://arxiv.org/pdf/1902.00751.pdf.
    """

    inv_adapter: Optional[str] = "nice"
    inv_adapter_reduction_factor: Optional[int] = 2


@dataclass(eq=False)
class ParallelConfig(AdapterConfig):
    """
    The parallel adapter architecture proposed by He et al. (2021). See https://arxiv.org/pdf/2110.04366.pdf.
    """

    original_ln_before: bool = False
    original_ln_after: bool = True
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = "relu"
    reduction_factor: Union[int, Mapping] = 2

    init_weights: str = "mam"
    is_parallel: bool = True
    scaling: Union[float, str] = 4.0


@dataclass(eq=False)
class PrefixTuningConfig(AdapterConfigBase):
    """
    The prefix tuning architecture proposed by Li & Liang (2021). See https://arxiv.org/pdf/2101.00190.pdf.
    """

    architecture: Optional[str] = "prefix_tuning"

    flat: bool = False
    prefix_length: int = 30
    bottleneck_size: int = 512
    non_linearity: str = "tanh"
    dropout: float = 0.0

    encoder_prefix: bool = True
    cross_prefix: bool = True
    leave_out: List[int] = field(default_factory=list)


class ConfigUnion(AdapterConfigBase):
    """
    Composes multiple adaptation method configurations into one. This class can be used to define complex adaptation
    method setups.
    """

    architecture: Optional[str] = "union"

    configs: List[AdapterConfigBase]

    def __init__(self, *configs: List[AdapterConfigBase]):
        self.validate(configs)
        self.configs = configs

    @staticmethod
    def validate(configs):
        """
        Performs simple validations of a list of configurations to check whether they can be combined to a common
        setup.

        Args:
            configs (List[AdapterConfigBase]): list of configs to check.

        Raises:
            TypeError: One of the configurations has a wrong type. ValueError: At least two given configurations
            conflict.
        """
        # perform single config checks
        for config in configs:
            if not isinstance(config, AdapterConfigBase):
                raise TypeError(f"{config} is not an instance of AdapterConfigBase")
            elif isinstance(config, ConfigUnion):
                raise TypeError(f"{config} of type {type(config)} is not supported in a config union.")
        # perform pairwise check
        for c_a, c_b in [(c_a, c_b) for i, c_a in enumerate(configs) for j, c_b in enumerate(configs) if i > j]:
            if c_a.architecture != c_b.architecture:
                continue
            elif c_a.architecture is None or c_a.architecture == "bottleneck":
                is_valid = c_a.mh_adapter != c_b.mh_adapter and c_a.output_adapter != c_b.output_adapter
                if not is_valid:
                    raise ValueError(f"{c_a} and {c_b} cannot be combined.")
                else:
                    continue
            # at this point, we know that the architectures are the same
            raise ValueError(f"{c_a} and {c_b} have the same adapter architecture and cannot be combined.")

    def __getitem__(self, key):
        i, k = key.split(".")
        return self.configs[int(i)][k]

    def __iter__(self):
        for i, c in enumerate(self.configs):
            for k in iter(c):
                yield f"{i}.{k}"

    def __len__(self):
        return sum([len(c) for c in self.configs])

    def __eq__(self, other):
        return all([c_a == c_b for c_a, c_b in zip(self.configs, other.configs)])

    def to_dict(self):
        return {"architecture": self.architecture, "configs": [c.to_dict() for c in self.configs]}

    def replace(self, **changes):
        return ConfigUnion(*[c.replace(**changes) for c in self.configs])

    @classmethod
    def from_dict(cls, config):
        if isinstance(config, AdapterConfigBase):
            return config

        configs = []
        for c in config["configs"]:
            config_class = cls._get_config_class(c)
            configs.append(config_class.from_dict(c))

        return cls(*configs)


class MAMConfig(ConfigUnion):
    """
    The Mix-And-Match adapter architecture proposed by He et al. (2021). See https://arxiv.org/pdf/2110.04366.pdf.
    """

    def __init__(self, prefix_tuning: Optional[PrefixTuningConfig] = None, adapter: Optional[AdapterConfig] = None):
        prefix_tuning = prefix_tuning or PrefixTuningConfig()
        adapter = adapter or ParallelConfig()

        assert isinstance(prefix_tuning, PrefixTuningConfig)
        assert isinstance(adapter, AdapterConfig)
        super().__init__(prefix_tuning, adapter)

    @property
    def prefix_tuning(self):
        return self[0]

    @property
    def adapter(self):
        return self[1]


ADAPTER_CONFIG_MAP = {
    "pfeiffer": PfeifferConfig(),
    "houlsby": HoulsbyConfig(),
    "pfeiffer+inv": PfeifferInvConfig(),
    "houlsby+inv": HoulsbyInvConfig(),
    "prefix": PrefixTuningConfig(),
    "prefix_flat": PrefixTuningConfig(flat=True),
    "parallel": ParallelConfig(),
    "scaled_parallel": ParallelConfig(scaling="learned"),
    "mam": MAMConfig(),
}

DEFAULT_ADAPTER_CONFIG = "pfeiffer"


class ModelAdaptersConfig(Collection):
    """This class manages the setup and configuration of adapter modules in a pre-trained model."""

    def __init__(self, **kwargs):
        adapters_list = kwargs.pop("adapters", {})
        # this is for backwards compability: in v1.x, self.adapters values had shape (<type>, <config_name>)
        adapters_list = dict(
            map(lambda t: (t[0], t[1][1] or t[1][0] if isinstance(t[1], tuple) else t[1]), adapters_list.items())
        )
        self.adapters: Mapping[str, str] = adapters_list
        self.config_map = kwargs.pop("config_map", {})

        self.fusions: Mapping[str, str] = kwargs.pop("fusions", {})
        self.fusion_config_map = kwargs.pop("fusion_config_map", {})

        # TODO-V2 Save this with config?
        self.active_setup: Optional[AdapterCompositionBlock] = None
        self.skip_layers = None

    def __contains__(self, item):
        return item in self.adapters.keys()

    def __iter__(self):
        return iter(self.adapters)

    def __len__(self):
        return len(self.adapters)

    def get(self, adapter_name: str) -> Optional[dict]:
        """
        Gets the config dictionary for a given adapter.

        Args:
            adapter_name (str): The name of the adapter.

        Returns:
            Mapping: The adapter configuration.
        """
        if adapter_name in self.adapters:
            config_name = self.adapters[adapter_name]
            if config_name in self.config_map:
                config = self.config_map.get(config_name, None)
            else:
                config = ADAPTER_CONFIG_MAP.get(config_name, None)
            if isinstance(config, str):
                config = ADAPTER_CONFIG_MAP[config]
        else:
            config = None
        return config

    def match(
        self,
        adapter_name: str,
        config_type: type,
        layer_idx: Optional[int] = None,
        location_key: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Tries to match the given criteria to an existing adapter. Return the adapter config if a match is found,
        otherwise None.
        """
        config = self.get(adapter_name)
        if config is None:
            return None

        if isinstance(config, config_type):
            leave_out = config.get("leave_out", [])
            if layer_idx is None or layer_idx not in leave_out:
                if location_key is None or config.get(location_key, False):
                    return config
        # if we have a config union, match with all child configs
        elif isinstance(config, ConfigUnion):
            for c in config.configs:
                if isinstance(c, config_type):
                    leave_out = c.get("leave_out", [])
                    if layer_idx is None or layer_idx not in leave_out:
                        if location_key is None or c.get(location_key, False):
                            return c

        return None

    def add(self, adapter_name: str, config: Optional[Union[str, dict]] = None):
        """
        Adds a new adapter of the name to the model config.

        Args:
            adapter_name (str): The name of the adapter.
            config (Optional[Union[str, dict]], optional): The adapter config. Defaults to None.
        """
        if adapter_name in self.adapters:
            raise ValueError(f"An adapter with the name '{adapter_name}' has already been added.")
        if config is None:
            config = DEFAULT_ADAPTER_CONFIG
        if isinstance(config, str):
            if config not in ADAPTER_CONFIG_MAP and config not in self.config_map:
                raise ValueError(f"Invalid adapter config identifier '{config}'.")
            config_name = config
        # if it's a dict, compute it's hash and add a new entry to the config map
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.config_map[config_name] = config
        else:
            raise ValueError("Invalid adapter config: {}".format(config))
        self.adapters[adapter_name] = config_name
        logger.info(f"Adding adapter '{adapter_name}'.")

    def get_fusion(self, fusion_name: Union[str, List[str]]) -> Optional[dict]:
        """
        Gets the config dictionary for a given AdapterFusion.

        Args:
            fusion_name (Union[str, List[str]]): The name of the AdapterFusion or the adapters to fuse.

        Returns:
            Optional[dict]: The AdapterFusion configuration.
        """
        if isinstance(fusion_name, list):
            fusion_name = ",".join(fusion_name)
        if fusion_name in self.fusions:
            config_name = self.fusions[fusion_name]
            if config_name in self.fusion_config_map:
                config = self.fusion_config_map.get(config_name, None)
            else:
                config = ADAPTERFUSION_CONFIG_MAP.get(config_name, None)
        else:
            config = None
        return config

    def add_fusion(self, fusion_name: Union[str, List[str]], config: Optional[Union[str, dict]] = None):
        """
        Adds a new AdapterFusion.

        Args:
            fusion_name (Union[str, List[str]]): The name of the AdapterFusion or the adapters to fuse.
            config (Optional[Union[str, dict]], optional): AdapterFusion config. Defaults to None.
        """
        if isinstance(fusion_name, list):
            fusion_name = ",".join(fusion_name)
        if fusion_name in self.fusions:
            raise ValueError(f"An AdapterFusion with the name '{fusion_name}' has already been added.")
        if config is None:
            config = DEFAULT_ADAPTERFUSION_CONFIG
        if isinstance(config, str):
            if config not in ADAPTERFUSION_CONFIG_MAP and config not in self.fusion_config_map:
                raise ValueError(f"Invalid AdapterFusion config identifier '{config}'.")
            config_name = config
        # if it's a dict, compute it's hash and add a new entry to the config map
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.fusion_config_map[config_name] = config
        else:
            raise ValueError("Invalid AdapterFusion config: {}".format(config))
        self.fusions[fusion_name] = config_name
        logger.info(f"Adding AdapterFusion '{fusion_name}'.")

    def common_config_value(self, adapter_names: list, attribute: str):
        """
        Checks whether all adapters in a list share the same config setting for a given attribute and returns the
        shared value.

        Args:
            adapter_names (list): The adapters to check.
            attribute (str): The config attribute to check.
        """
        common_value = None
        for i, name in enumerate(adapter_names):
            config = self.get(name)
            if not config:
                raise ValueError(
                    f"No adapter with name '{name}' found. Make sure that an adapter with this name is loaded."
                )
            config_value = config.get(attribute, None)
            if i > 0 and config_value != common_value:
                raise ValueError(f"All given adapters must define the same value for config attribute {attribute}.")
            common_value = config_value
        return common_value

    def to_dict(self):
        output_dict = {}
        output_dict["adapters"] = copy.deepcopy(self.adapters)
        output_dict["config_map"] = {}
        for k, v in self.config_map.items():
            if isinstance(v, AdapterConfigBase):
                output_dict["config_map"][k] = v.to_dict()
            else:
                output_dict["config_map"][k] = copy.deepcopy(v)
        output_dict["fusions"] = copy.deepcopy(self.fusions)
        output_dict["fusion_config_map"] = {}
        for k, v in self.fusion_config_map.items():
            if isinstance(v, AdapterConfigBase):
                output_dict["fusion_config_map"][k] = v.to_dict()
            else:
                output_dict["fusion_config_map"][k] = copy.deepcopy(v)
        return output_dict


def build_full_config(adapter_config, model_config, save_id2label=False, **kwargs):
    config_dict = {
        "model_type": model_config.model_type,
        # some models such as encoder-decoder don't have a model-wide hidden size
        "hidden_size": getattr(model_config, "hidden_size", None),
    }
    config_dict.update(kwargs)
    if not hasattr(model_config, "prediction_heads") and save_id2label:
        config_dict["label2id"] = model_config.label2id
    if isinstance(adapter_config, AdapterConfigBase):
        config_dict["config"] = adapter_config.to_dict()
    else:
        config_dict["config"] = adapter_config
    return config_dict


@dataclass(eq=False)
class AdapterFusionConfig(AdapterConfigBase):
    """Base class that models the architecture of an adapter fusion layer."""

    key: bool
    query: bool
    value: bool
    query_before_ln: bool
    regularization: bool
    residual_before: bool
    temperature: bool
    value_before_softmax: bool
    value_initialized: str

    @classmethod
    def load(cls, config: Union[dict, str], **kwargs):
        """
        Loads a given adapter fusion configuration specifier into a full AdapterFusionConfig instance.

        Args:
            config (Union[dict, str]): The configuration to load. Can be either:

                - a dictionary representing the full config
                - an identifier string available in ADAPTERFUSION_CONFIG_MAP
                - the path to a file containing a full adapter fusion configuration

        Returns:
            dict: The resolved adapter fusion configuration dictionary.
        """
        # currently storing AdapterFusion weights on AdapterHub is not supported.
        config_dict = resolve_adapter_config(config, local_map=ADAPTERFUSION_CONFIG_MAP, try_loading_from_hub=False)
        # convert back to dict to allow attr overrides
        if isinstance(config_dict, AdapterFusionConfig):
            config_dict = config_dict.to_dict()
        config_dict.update(kwargs)
        return AdapterFusionConfig.from_dict(config_dict)


@dataclass(eq=False)
class StaticAdapterFusionConfig(AdapterFusionConfig):
    """
    Static version of adapter fusion without a value matrix. Described in https://arxiv.org/pdf/2005.00247.pdf.
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


@dataclass(eq=False)
class DynamicAdapterFusionConfig(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. Described in
    https://arxiv.org/pdf/2005.00247.pdf.
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
