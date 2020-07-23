import json
import logging
from abc import ABC, abstractmethod
from os import mkdir
from os.path import exists, isdir, isfile, join
from typing import Callable, List, Mapping, Optional, Tuple, Union

import torch

from .adapter_config import (
    ADAPTERFUSION_CONFIG_MAP,
    DEFAULT_ADAPTERFUSION_CONFIG,
    AdapterConfig,
    AdapterFusionConfig,
    AdapterType,
    build_full_config,
    get_adapter_config_hash,
)
from .adapter_utils import (
    ADAPTERFUSION_CONFIG_NAME,
    ADAPTERFUSION_WEIGHTS_NAME,
    CONFIG_NAME,
    HEAD_CONFIG_NAME,
    HEAD_WEIGHTS_NAME,
    WEIGHTS_NAME,
    inherit_doc,
    resolve_adapter_path,
)


logger = logging.getLogger(__name__)


class WeightsLoaderHelper:
    """
    A class providing helper methods for saving and loading module weights.
    """

    def __init__(self, model, weights_name, config_name):
        self.model = model
        self.weights_name = weights_name
        self.config_name = config_name

    def state_dict(self, filter_func):
        return {k: v for (k, v) in self.model.state_dict().items() if filter_func(k)}

    def rename_state_dict(self, state_dict, rename_func):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = rename_func(k)
            new_state_dict[new_k] = v
        return new_state_dict

    def save_weights_config(self, save_directory, config, meta_dict=None):
        # add meta information if given
        if meta_dict:
            for k, v in meta_dict.items():
                if k not in config:
                    config[k] = v
        # save to file system
        output_config_file = join(save_directory, self.config_name)
        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        logger.info("Configuration saved in {}".format(output_config_file))

    def save_weights(self, save_directory, filter_func):
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), "Saving path should be a directory where the module weights can be saved."

        # Get the state of all adapter modules for this task
        state_dict = self.state_dict(filter_func)
        # Save the adapter weights
        output_file = join(save_directory, self.weights_name)
        torch.save(state_dict, output_file)
        logger.info("Module weights saved in {}".format(output_file))

    def load_weights_config(self, save_directory):
        config_file = join(save_directory, self.config_name)
        logger.info("Loading module configuration from {}".format(config_file))
        # Load the config
        with open(config_file, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        return loaded_config

    @staticmethod
    def _load_module_state_dict(module, state_dict, start_prefix=""):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(module, prefix=start_prefix)

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    module.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return missing_keys, unexpected_keys

    def load_weights(self, save_directory, filter_func, rename_func=None, loading_info=None, in_base_model=False):
        weights_file = join(save_directory, self.weights_name)
        # Load the weights of the adapter
        try:
            state_dict = torch.load(weights_file, map_location="cpu")
        except Exception:
            raise OSError("Unable to load weights from pytorch checkpoint file. ")

        # Rename weights if needed
        if rename_func:
            state_dict = self.rename_state_dict(state_dict, rename_func)

        logger.info("Loading module weights from {}".format(weights_file))

        # Add the weights to the model
        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = self.model
        has_prefix_module = any(s.startswith(self.model.base_model_prefix) for s in state_dict.keys())
        if not hasattr(self.model, self.model.base_model_prefix) and has_prefix_module:
            start_prefix = self.model.base_model_prefix + "."
        if in_base_model and hasattr(self.model, self.model.base_model_prefix) and not has_prefix_module:
            model_to_load = self.model.base_model

        missing_keys, unexpected_keys = self._load_module_state_dict(
            model_to_load, state_dict, start_prefix=start_prefix
        )

        missing_keys = [k for k in missing_keys if filter_func(k)]
        if len(missing_keys) > 0:
            logger.info(
                "Some module weights could not be found in loaded weights file: {}".format(", ".join(missing_keys))
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Some weights of the state_dict could not be loaded into model: {}".format(", ".join(unexpected_keys))
            )

        if isinstance(loading_info, dict):
            if "missing_keys" not in loading_info:
                loading_info["missing_keys"] = []
            if "unexpected_keys" not in loading_info:
                loading_info["unexpected_keys"] = []
            loading_info["missing_keys"].extend(missing_keys)
            loading_info["unexpected_keys"].extend(unexpected_keys)

        return missing_keys, unexpected_keys


class WeightsLoader(ABC):
    """
    An abstract class providing basic methods for saving and loading weights of a model.
    Extend this class to build custom module weight loaders.
    """

    def __init__(self, model, weights_name, config_name):
        self.model = model
        self.weights_helper = WeightsLoaderHelper(model, weights_name, config_name)

    @abstractmethod
    def filter_func(self, name: str) -> Callable[[str], bool]:
        """The callable returned by this method is used to extract the module weights to be saved or loaded
        based on their names.

        Args:
            name (str): An identifier of the weights to be saved.

        Returns:
            Callable[str, bool]: A function that takes the fully qualified name of a module parameter and returns
                                a boolean value that specifies whether this parameter should be extracted.
        """
        pass

    @abstractmethod
    def rename_func(self, old_name: str, new_name: str) -> Callable[[str], str]:
        """The callable returned by this method is used to optionally rename the module weights after loading.

        Args:
            old_name (str): The string identifier of the weights as loaded from file.
            new_name (str): The new string identifier to which the weights should be renamed.

        Returns:
            Callable[str, str]: A function that takes the fully qualified name of a module parameter and returns
                                a new fully qualified name.
        """
        pass

    def save(self, save_directory, name, **kwargs):
        """Saves the module config and weights into the given directory.
        Override this method for additional saving actions.

        Args:
            save_directory (str): The directory to save the weights in.
            name (str): An identifier of the weights to be saved. The details are specified by the implementor.
        """
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(
                save_directory
            ), "Saving path should be a directory where weights and configuration can be saved."

        config_dict = build_full_config(
            None,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )
        meta_dict = kwargs.pop("meta_dict", None)

        # Save the adapter configuration
        self.weights_helper.save_weights_config(save_directory, config_dict, meta_dict=meta_dict)

        # Save adapter weights
        filter_func = self.filter_func(name)
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(self, save_directory, load_as=None, loading_info=None, **kwargs) -> Tuple[str, str]:
        """Loads the module weights from the given directory.
        Override this method for additional loading actions. If adding the loaded weights
        to the model passed to the loader class requires adding additional modules, this method should also perform the
        architectural changes to the model.

        Args:
            save_directory (str): The directory from where to load the weights.
            load_as (str, optional): Load the weights with this name. Defaults to None.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
                             and the name of the loaded weights.
        """
        if not exists(join(save_directory, self.weights_helper.weights_name)):
            raise ValueError("Loading path should be a directory where the weights are saved.")

        # Load config
        config = self.weights_helper.load_weights_config(save_directory)

        # Load head weights
        filter_func = self.filter_func(config["name"])
        if load_as:
            rename_func = self.rename_func(config["name"], load_as)
        else:
            rename_func = None
        self.weights_helper.load_weights(
            save_directory, filter_func, rename_func=rename_func, loading_info=loading_info
        )

        return save_directory, load_as or config["name"]


class AdapterLoader(WeightsLoader):
    """
    A class providing methods for saving and loading adapter modules from the Hub, the filesystem or a remote url.

    Model classes passed to this loader must implement the `ModelAdaptersMixin` class.
    """

    def __init__(self, model, adapter_type=None):
        super().__init__(model, WEIGHTS_NAME, CONFIG_NAME)
        self.adapter_type = adapter_type

    @property
    def config(self):
        return self.model.config.adapters.get_config(self.adapter_type)

    def filter_func(self, adapter_name):
        if self.adapter_type == AdapterType.text_lang:
            return (
                lambda x: "{}_adapters.{}".format(self.adapter_type, adapter_name) in x
                or "invertible_lang_adapters.{}".format(adapter_name) in x
            )
        elif AdapterType.has(self.adapter_type):
            return lambda x: "{}_adapters.{}".format(self.adapter_type, adapter_name) in x
        else:
            raise ValueError("Invalid adapter type {}".format(self.adapter_type))

    def rename_func(self, old_name, new_name):
        return lambda k: k.replace("_adapters.{}".format(old_name), "_adapters.{}".format(new_name))

    def save(self, save_directory, name, meta_dict=None):
        """Saves an adapter and its configuration file to a directory, so that it can be reloaded
        using the `load()` method.

        Args:
            save_directory (str): a path to a directory where the adapter will be saved
            task_name (str): the name of the adapter to be saved
        """
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(
                save_directory
            ), "Saving path should be a directory where adapter and configuration can be saved."
        assert (
            name in self.model.config.adapters.adapters
        ), "No adapter of this type with the given name is part of this model."

        adapter_config, adapter_type = self.model.config.adapters.get(name, return_type=True)
        if self.adapter_type:
            assert adapter_type == self.adapter_type, "Saved adapter has to be a {} adapter.".format(self.adapter_type)
        else:
            self.adapter_type = adapter_type

        config_dict = build_full_config(
            adapter_config,
            self.model.config,
            type=adapter_type,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )

        # Save the adapter configuration
        self.weights_helper.save_weights_config(save_directory, config_dict, meta_dict=meta_dict)

        # Save adapter weights
        filter_func = self.filter_func(config_dict["name"])
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(
        self,
        adapter_name_or_path,
        config=None,
        version=None,
        model_name=None,
        load_as=None,
        loading_info=None,
        **kwargs
    ):
        """Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (str, optional): The requested configuration of the adapter.
            version (str, optional): The version of the adapter to be loaded.
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
             saved will be used.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
                             and the name of the loaded weights.
        """
        # use: given adapter config (can be string) > default config of this type > global default config
        config = config or self.config
        requested_config = AdapterConfig.load(config) if config else None
        # Resolve the weights to be loaded based on the given identifier and the current adapter config
        model_name = self.model.model_name or model_name
        resolved_folder = resolve_adapter_path(
            adapter_name_or_path,
            self.adapter_type,
            model_name,
            adapter_config=requested_config,
            version=version,
            **kwargs,
        )

        # Load config of adapter
        config = self.weights_helper.load_weights_config(resolved_folder)
        if self.adapter_type:
            assert config["type"] == self.adapter_type, "Loaded adapter has to be a {} adapter.".format(
                self.adapter_type
            )
        else:
            self.adapter_type = config["type"]

        adapter_name = load_as or config["name"]
        # If the adapter is not part of the model, add it
        if adapter_name not in self.model.config.adapters.adapters:
            self.model.add_adapter(adapter_name, config["type"], config=config["config"])
        else:
            logger.warning("Overwriting existing adapter '{}'.".format(adapter_name))

        # Load adapter weights
        filter_func = self.filter_func(adapter_name)
        rename_func = self.rename_func(config["name"], adapter_name)
        self.weights_helper.load_weights(
            resolved_folder, filter_func, rename_func=rename_func, loading_info=loading_info, in_base_model=True
        )

        return resolved_folder, adapter_name


class AdapterFusionLoader(WeightsLoader):
    """
    A class providing methods for saving and loading AdapterFusion modules from the file system.

    """

    def __init__(self, model, error_on_missing=True):
        super().__init__(model, ADAPTERFUSION_WEIGHTS_NAME, ADAPTERFUSION_CONFIG_NAME)
        self.error_on_missing = error_on_missing

    def filter_func(self, adapter_fusion_name):
        return lambda x: "adapter_fusion_layer.{}".format(adapter_fusion_name) in x

    def rename_func(self, old_name, new_name):
        return lambda k: k.replace(
            "adapter_fusion_layer.{}".format(old_name), "adapter_fusion_layer.{}".format(new_name)
        )

    def save(self, save_directory: str, name: str):
        """Saves a AdapterFusion module into the given directory.

        Args:
            save_directory (str): The directory to save the weights in.
            name (str, optional): The AdapterFusion name.
        """

        if hasattr(self.model.config, "adapter_fusion_models"):
            if name not in self.model.config.adapter_fusion_models:
                if self.error_on_missing:
                    raise ValueError(f"Unknown AdapterFusion '{name}'.")
                else:
                    logger.debug(f"No AdapterFusion with name '{name}' available.")
                    return

        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), "Saving path should be a directory where the head can be saved."

        adapter_fusion_config = self.model.config.adapter_fusion

        # Save the adapter fusion configuration
        config_dict = build_full_config(
            adapter_fusion_config,
            self.model.config,
            name=name,
            model_name=self.model.model_name,
            model_class=self.model.__class__.__name__,
        )
        self.weights_helper.save_weights_config(save_directory, config_dict)

        # Save head weights
        filter_func = self.filter_func(name)
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(self, save_directory, load_as=None, loading_info=None):
        """Loads a AdapterFusion module from the given directory.

        Args:
            save_directory (str): The directory from where to load the weights.
            load_as (str, optional): Load the weights with this name. Defaults to None.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
                             and the name of the loaded weights.
        """
        if not exists(join(save_directory, ADAPTERFUSION_WEIGHTS_NAME)):
            if self.error_on_missing:
                raise ValueError("Loading path should be a directory where AdapterFusion is saved.")
            else:
                logger.debug("No matching adapter fusion found in '{}'".format(save_directory))
                return None, None

        config = self.weights_helper.load_weights_config(save_directory)
        if not hasattr(self.model.config, "adapter_fusion_models"):
            self.model.config.adapter_fusion_models = []

        adapter_fusion_name = load_as or config["name"]
        if adapter_fusion_name in self.model.config.adapter_fusion_models:
            logger.warning("Overwriting existing adapter fusion module '{}'".format(adapter_fusion_name))
        self.model.add_fusion(adapter_fusion_name, config["config"])

        # Load AdapterFusion weights
        filter_func = self.filter_func(adapter_fusion_name)
        if load_as:
            rename_func = self.rename_func(config["name"], load_as)
        else:
            rename_func = None
        self.weights_helper.load_weights(
            save_directory, filter_func, rename_func=rename_func, loading_info=loading_info
        )

        return save_directory, adapter_fusion_name


class PredictionHeadLoader(WeightsLoader):
    """
    A class providing methods for saving and loading prediction head modules from the file system.

    Model classes supporting configurable head modules via config files should provide
    a prediction head config at `model.config.prediction_heads` and a method `add_prediction_head(head_name, config)`.
    """

    def __init__(self, model, error_on_missing=True):
        super().__init__(model, HEAD_WEIGHTS_NAME, HEAD_CONFIG_NAME)
        self.error_on_missing = error_on_missing

    def filter_func(self, head_name):
        if head_name:
            return lambda x: not x.startswith(self.model.base_model_prefix) and "heads.{}".format(head_name) in x
        else:
            return lambda x: not x.startswith(self.model.base_model_prefix)

    def rename_func(self, old_name, new_name):
        return lambda k: k.replace("heads.{}".format(old_name), "heads.{}".format(new_name))

    def save(self, save_directory: str, name: str = None):
        """Saves a prediction head module into the given directory.

        Args:
            save_directory (str): The directory to save the weights in.
            name (str, optional): The prediction head name.
        """

        if name:
            if hasattr(self.model.config, "prediction_heads"):
                if name not in self.model.config.prediction_heads:
                    if self.error_on_missing:
                        raise ValueError(f"Unknown head_name '{name}'.")
                    else:
                        logger.debug(f"No prediction head with name '{name}' available.")
                        return
            else:
                # we haven't found a prediction head configuration, so we assume there is only one (unnamed) head
                # (e.g. this is the case if we use a 'classic' Hf model with head)
                # -> ignore the name and go on
                name = None
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), "Saving path should be a directory where the head can be saved."

        # if we use a custom head, save it
        if name and hasattr(self.model.config, "prediction_heads"):
            head_config = self.model.config.prediction_heads[name]
        else:
            head_config = None

        # Save the adapter configuration
        config_dict = build_full_config(
            head_config,
            self.model.config,
            name=name,
            model_name=self.model.model_name,
            model_class=self.model.__class__.__name__,
        )
        self.weights_helper.save_weights_config(save_directory, config_dict)

        # Save head weights
        filter_func = self.filter_func(name)
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(self, save_directory, load_as=None, loading_info=None):
        """Loads a prediction head module from the given directory.

        Args:
            save_directory (str): The directory from where to load the weights.
            load_as (str, optional): Load the weights with this name. Defaults to None.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
                             and the name of the loaded weights.
        """
        if not exists(join(save_directory, HEAD_WEIGHTS_NAME)):
            if self.error_on_missing:
                raise ValueError("Loading path should be a directory where the head is saved.")
            else:
                logger.info("No matching prediction head found in '{}'".format(save_directory))
                return None, None

        head_name = None

        # Load head config if available - otherwise just blindly try to load the weights
        if isfile(join(save_directory, HEAD_CONFIG_NAME)):
            config = self.weights_helper.load_weights_config(save_directory)
            # make sure that the model class of the loaded head matches the current class
            if self.model.__class__.__name__ != config["model_class"]:
                if self.error_on_missing:
                    raise ValueError(
                        f"Model class '{config['model_class']}' of found prediction head does not match current "
                        f"model class."
                    )
                else:
                    logger.debug("No matching prediction head found in '{}'".format(save_directory))
                    return None, None
            if hasattr(self.model.config, "prediction_heads"):
                head_name = load_as or config["name"]
                if head_name in self.model.config.prediction_heads:
                    logger.warning("Overwriting existing head '{}'".format(head_name))
                self.model.add_prediction_head(head_name, config["config"], overwrite_ok=True)

        # Load head weights
        filter_func = self.filter_func(head_name)
        if load_as:
            rename_func = self.rename_func(config["name"], load_as)
        else:
            rename_func = None
        self.weights_helper.load_weights(
            save_directory, filter_func, rename_func=rename_func, loading_info=loading_info
        )

        return save_directory, head_name


class ModelAdaptersMixin(ABC):
    """Mixin for transformer models adding support for loading/ saving adapters."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model_name = None

    # These methods have to be implemented by every deriving class:

    @abstractmethod
    def add_adapter(self, adapter_name: str, adapter_type: AdapterType, config=None):
        """Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            adapter_type (AdapterType): The adapter type.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:
                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        pass

    @abstractmethod
    def train_adapter(self, adapter_names: list):
        """Sets the model in mode for training the given type of adapter.
        """
        pass

    def has_adapters(self, adapter_type=None):
        if not adapter_type:
            return len(self.config.adapters.adapters) > 0
        else:
            return len(self.config.adapters.adapter_list(adapter_type)) > 0

    def set_adapter_config(self, adapter_type: AdapterType, adapter_config):
        """Sets the adapter configuration of the specified adapter type.

        Args:
            adapter_type (AdapterType): The adapter type.
            adapter_config (str or dict): adapter configuration, can be either:
                - a string identifying a pre-defined adapter configuration
                - a dictionary representing the adapter configuration
                - the path to a file containing the adapter configuration
        """
        if AdapterType.has(adapter_type):
            self.config.adapters.set_config(adapter_type, adapter_config)
        else:
            raise ValueError("Invalid adapter type {}".format(adapter_type))

    def set_adapter_fusion_config(self, adapter_fusion_config, kwargs={}):
        """Sets the adapter fusion configuration.

        Args:
            adapter_fusion_config (str or dict): adapter fusion configuration, can be either:
                - a string identifying a pre-defined adapter fusion configuration
                - a dictionary representing the adapter fusion configuration
                - the path to a file containing the adapter fusion configuration
        """
        if isinstance(adapter_fusion_config, str) and adapter_fusion_config in ADAPTERFUSION_CONFIG_MAP:
            self.config.adapter_fusion = AdapterFusionConfig.load(adapter_fusion_config, **kwargs)
            # ADAPTERFUSION_CONFIG_MAP[adapter_fusion_config](**kwargs).to_dict()
        elif isinstance(adapter_fusion_config, Mapping):
            self.config.adapter_fusion = adapter_fusion_config
        else:
            raise ValueError("Invalid adapter type {}".format(adapter_fusion_config))

    def add_fusion(self, adapter_names, adapter_fusion_config=None, kwargs={}):
        """Adds AdapterFusion to the model with alll the necessary configurations and weight initializations

        Args:
            adapter_names: a list of adapter names which should be fused
            adapter_fusion_config (str or dict): adapter fusion configuration, can be either:
                - a string identifying a pre-defined adapter fusion configuration
                - a dictionary representing the adapter fusion configuration
                - the path to a file containing the adapter fusion configuration
            kwargs: dictionary items for values which should be overwritten in the default AdapterFusion configuration

        Returns:

        """
        if not hasattr(self.config, "adapter_fusion"):
            if adapter_fusion_config is not None:
                self.set_adapter_fusion_config(adapter_fusion_config, kwargs)
            else:
                self.set_adapter_fusion_config(DEFAULT_ADAPTERFUSION_CONFIG)
        elif hasattr(self.config, "adapter_fusion") and adapter_fusion_config is not None:
            raise Warning("An AdapterFusion config has already been set and will NOT be overwritten")

        if not hasattr(self.config, "adapter_fusion_models"):
            self.config.adapter_fusion_models = []
        if isinstance(adapter_names, list):
            adapter_fusion_name = ",".join(adapter_names)
        else:
            adapter_fusion_name = adapter_names
        if adapter_fusion_name not in self.config.adapter_fusion_models:
            self.config.adapter_fusion_models.append(adapter_fusion_name)
            self.base_model.add_fusion_layer(adapter_names)

    def save_adapter(
        self,
        save_directory: str,
        adapter_name: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """Saves an adapter and its configuration file to a directory so that it can be shared
        or reloaded using `load_adapter()`.

        Args:
            save_directory (str): Path to a directory where the adapter should be saved.
            adapter_name (str): Name of the adapter to be saved.

        Raises:
            ValueError: If the given adapter name is invalid.
        """
        adapter_type = self.config.adapters.get_type(adapter_name)
        if adapter_type:
            loader = AdapterLoader(self, adapter_type)
            loader.save(save_directory, adapter_name, meta_dict)
            # save additional custom weights
            if custom_weights_loaders:
                for weights_loader in custom_weights_loaders:
                    weights_loader.save(save_directory, adapter_name)
        else:
            raise ValueError("Could not resolve '{}' to a valid adapter name.".format(adapter_name))

    def save_adapter_fusion(
        self, save_directory: str, adapter_names: list, custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """Saves an adapter and its configuration file to a directory so that it can be shared
        or reloaded using `load_adapter()`.

        Args:
            save_directory (str): Path to a directory where the adapter should be saved.
            adapter_name (str): Name of the adapter to be saved.

        Raises:
            ValueError: If the given adapter name is invalid.
        """

        loader = AdapterFusionLoader(self)
        loader.save(save_directory, adapter_names)
        # save additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.save(save_directory, adapter_names)

    def load_adapter(
        self,
        adapter_name_or_path: str,
        adapter_type: AdapterType = None,
        config: Union[dict, str] = None,
        version: str = None,
        model_name: str = None,
        load_as: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        **kwargs
    ) -> str:
        """Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            adapter_type (AdapterType, optional): The type of adapter to be loaded. If not specified, text_task will be
                    used for adapters loaded from the Hub.
            config (dict or str, optional): The requested configuration of the adapter.
                If not specified, will be either:
                - the default adapter config for the requested adapter if specified
                - the global default adapter config
            version (str, optional): The version of the adapter to be loaded.
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
                    saved will be used.

        Returns:
            str: The name with which the adapter was added to the model.
        """
        if AdapterType.has(adapter_type) or not adapter_type:
            loader = AdapterLoader(self, adapter_type)
            load_dir, load_name = loader.load(adapter_name_or_path, config, version, model_name, load_as, **kwargs)
            # load additional custom weights
            if custom_weights_loaders:
                for weights_loader in custom_weights_loaders:
                    weights_loader.load(load_dir, load_as=load_as, loading_info=kwargs.get("loading_info", None))
            return load_name
        else:
            raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def load_adapter_fusion(
        self,
        adapter_fusion_name_or_path: str,
        load_as: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        **kwargs
    ) -> str:
        """Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_fusion_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter fusion module to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (dict or str, optional): The requested configuration of the adapter fusion.
                If not specified, will be either:
                - the default adapter config for the requested adapter fusion if specified
                - the global default adapter fusion config
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
                    saved will be used.

        Returns:
            str: The name with which the adapter was added to the model.
        """

        loader = AdapterFusionLoader(self)
        load_dir, load_name = loader.load(adapter_fusion_name_or_path, load_as)
        # load additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(load_dir, load_as=load_as, loading_info=kwargs.get("loading_info", None))
        return load_name

    def save_all_adapters(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """Saves all adapters of this model together with their configuration
        to subfolders of the given location.

        Args:
            save_directory (str): Path to a directory where the adapters should be saved.
        """
        for name in self.config.adapters.adapters:
            adapter_config, adapter_type = self.config.adapters.get(name, return_type=True)
            h = get_adapter_config_hash(adapter_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({"config_id": h})
            else:
                meta_dict = {"config_id": h}
            self.save_adapter(save_path, name, meta_dict=meta_dict, custom_weights_loaders=custom_weights_loaders)

    def save_all_adapter_fusions(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """Saves all adapters of this model together with their configuration
        to subfolders of the given location.

        Args:
            save_directory (str): Path to a directory where the adapters should be saved.
        """
        for name in self.config.adapter_fusion_models:
            adapter_fusion_config = self.config.adapter_fusion
            h = get_adapter_config_hash(adapter_fusion_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({"config_id": h})
            else:
                meta_dict = {"config_id": h}
            self.save_adapter_fusion(save_path, name, custom_weights_loaders=custom_weights_loaders)

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model.
        """
        # first freeze/ unfreeze all model weights
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
        self.model_freezed = freeze


@inherit_doc
class ModelWithHeadsAdaptersMixin(ModelAdaptersMixin):
    """Mixin adding support for loading/ saving adapters to transformer models with head(s)."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType, config=None):
        """Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            adapter_type (AdapterType): The adapter type.
            config (str or dict, optional): The adapter configuration, can be either:
                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        self.base_model.add_adapter(adapter_name, adapter_type, config)

    def train_adapter(self, adapter_names: list):
        """Sets the model in mode for training the given type of adapter."""
        self.base_model.train_adapter(adapter_names)

    def train_fusion(self, adapter_names: list):
        """Sets the model in mode for training of adapter fusion determined by a list of adapter names."""
        self.base_model.train_fusion(adapter_names)

    def save_head(self, save_directory: str, head_name: str = None):
        loader = PredictionHeadLoader(self)
        loader.save(save_directory, name=head_name)

    def load_head(self, save_directory, load_as=None):
        loader = PredictionHeadLoader(self)
        return loader.load(save_directory, load_as=load_as)

    def save_adapter(
        self,
        save_directory: str,
        adapter_name: str,
        with_head: bool = True,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            if not any([isinstance(o, PredictionHeadLoader) for o in custom_weights_loaders]):
                custom_weights_loaders.append(PredictionHeadLoader(self, error_on_missing=False))
        super().save_adapter(
            save_directory, adapter_name, meta_dict=meta_dict, custom_weights_loaders=custom_weights_loaders,
        )

    def load_adapter(
        self,
        adapter_name_or_path: str,
        adapter_type: AdapterType = None,
        config: Union[dict, str] = None,
        version: str = None,
        model_name: str = None,
        load_as: str = None,
        with_head: bool = True,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        **kwargs
    ) -> str:
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            custom_weights_loaders.append(PredictionHeadLoader(self, error_on_missing=False))
        return super().load_adapter(
            adapter_name_or_path,
            adapter_type=adapter_type,
            config=config,
            version=version,
            model_name=model_name,
            load_as=load_as,
            custom_weights_loaders=custom_weights_loaders,
            **kwargs,
        )

    def save_all_adapters(
        self,
        save_directory: str,
        with_head: bool = True,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            custom_weights_loaders.append(PredictionHeadLoader(self, error_on_missing=False))
        super().save_all_adapters(
            save_directory, meta_dict=meta_dict, custom_weights_loaders=custom_weights_loaders,
        )
