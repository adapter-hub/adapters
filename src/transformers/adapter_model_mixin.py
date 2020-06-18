from abc import ABC, abstractmethod
import logging
from os import mkdir
from os.path import isdir, isfile, join, exists
import torch
import json
from .adapter_utils import (
    CONFIG_NAME, HEAD_CONFIG_NAME, WEIGHTS_NAME, HEAD_WEIGHTS_NAME,
    resolve_adapter_config,
    resolve_adapter_path,
)
from .adapter_config import (
    DEFAULT_ADAPTER_CONFIG,
    AdapterType,
    build_full_config,
    get_adapter_config_hash,
)


logger = logging.getLogger(__name__)


class WeightsLoader:
    """ A class providing methods for saving and loading weights of a model."""

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
        with open(config_file, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        return loaded_config

    def load_weights(self, save_directory, filter_func, rename_func=None):
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
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        missing_keys = [k for k in missing_keys if filter_func(k)]
        if len(missing_keys) > 0:
            logger.warn(
                "Some module weights could not be found in loaded weights file: {}".format(
                    ', '.join(missing_keys))
            )
        if len(unexpected_keys) > 0:
            logger.warn(
                "Some weights of the state_dict could not be loaded into model: {}".format(
                    ', '.join(unexpected_keys))
            )

        return missing_keys, unexpected_keys


class AdapterLoader(WeightsLoader):
    """A class providing methods for saving and loading adapter modules from the Hub, the filesystem or a remote url."""

    def __init__(self, model, adapter_type=None):
        super().__init__(model, WEIGHTS_NAME, CONFIG_NAME)
        self.adapter_type = adapter_type

    @property
    def config(self):
        return self.model.config.adapters.get_config(self.adapter_type)

    def _get_params_check_func(self, adapter_type, adapter_name):
        if adapter_type == AdapterType.text_lang:
            return lambda x: '{}_adapters.{}'.format(adapter_type, adapter_name) in x \
                or 'invertible_lang_adapters.{}'.format(adapter_name) in x
        elif AdapterType.has(adapter_type):
            return lambda x: '{}_adapters.{}'.format(adapter_type, adapter_name) in x
        else:
            raise ValueError("Invalid adapter type {}".format(adapter_type))

    def _rename_func(self, old_name, new_name):
        return lambda k: k.replace('_adapters.{}'.format(old_name), '_adapters.{}'.format(new_name))

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
            assert isdir(save_directory), "Saving path should be a directory where adapter and configuration can be saved."
        assert name in self.model.config.adapters.adapters, "No adapter of this type with the given name is part of this model."

        adapter_config, adapter_type = self.model.config.adapters.get(name, return_type=True)
        config_dict = build_full_config(
            adapter_config, self.model.config, type=adapter_type,
            model_name=self.model.model_name, name=name
        )
        # add meta information if given
        if meta_dict:
            for k, v in meta_dict.items():
                if k not in config_dict:
                    config_dict[k] = v

        # Save the adapter configuration
        output_config_file = join(save_directory, CONFIG_NAME)
        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
        logger.info("Configuration saved in {}".format(output_config_file))

        # Save adapter weights
        filter_func = self._get_params_check_func(adapter_type, config_dict['name'])
        self.save_weights(save_directory, filter_func)

    def load(self, adapter_name_or_path, config=None,
             version=None, model_name=None, load_as=None, **kwargs):
        """Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (str, optional): The requested configuration of the adapter.
            version (int, optional): The version of the adapter to be loaded.
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was saved will be used.
        """
        # use: given adapter config (can be string) > default config of this type > global default config
        requested_config = resolve_adapter_config(
            config or self.config or DEFAULT_ADAPTER_CONFIG
        )
        # Resolve the weights to be loaded based on the given identifier and the current adapter config
        model_name = self.model.model_name or model_name
        resolved_folder = resolve_adapter_path(
            adapter_name_or_path, requested_config, self.adapter_type, model_name, version, **kwargs
        )

        # Load config of adapter
        config = self.load_weights_config(resolved_folder)
        if self.adapter_type:
            assert config['type'] == self.adapter_type, "Loaded adapter has to be a {} adapter.".format(self.adapter_type)

        adapter_name = load_as or config['name']
        # If the adapter is not part of the model, add it
        if adapter_name not in self.model.config.adapters.adapters:
            self.model.add_adapter(adapter_name, config['type'], config=config['config'])
        else:
            logger.warn("Overwriting existing adapter '{}'.".format(adapter_name))

        # Load adapter weights
        filter_func = self._get_params_check_func(config['type'], config['name'])
        rename_func = self._rename_func(config['name'], adapter_name)
        self.load_weights(resolved_folder, filter_func, rename_func=rename_func)

        return adapter_name


class PredictionHeadLoader(WeightsLoader):
    """A class providing methods for saving and loading prediction head modules from the file system."""

    def __init__(self, model):
        super().__init__(model, HEAD_WEIGHTS_NAME, HEAD_CONFIG_NAME)

    def _filter_func(self, head_name):
        if head_name:
            return lambda x: not x.startswith(self.model.base_model_prefix) \
                and 'heads.{}'.format(head_name) in x
        else:
            return lambda x: not x.startswith(self.model.base_model_prefix)

    def _rename_func(self, old_name, new_name):
        return lambda k: k.replace('heads.{}'.format(old_name), 'heads.{}'.format(new_name))

    def save(self, save_directory: str, head_name: str = None):
        if head_name and head_name not in self.model.config.prediction_heads:
            raise ValueError(f"Unknown head_name '{head_name}'.")
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), "Saving path should be a directory where the head can be saved."

        # if we use a custom head, save it
        if head_name and hasattr(self.model.config, 'prediction_heads'):
            head_config = self.model.config.prediction_heads[head_name]
            config_dict = build_full_config(
                head_config, self.model.config,
                model_name=self.model.model_name, name=head_name
            )
            # Save the adapter configuration
            output_config_file = join(save_directory, HEAD_CONFIG_NAME)
            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
            logger.info("Head configuration saved in {}".format(output_config_file))

        # Save head weights
        filter_func = self._filter_func(head_name)
        self.save_weights(save_directory, filter_func)

    def load(self, save_directory, load_as=None):
        if not exists(join(save_directory, HEAD_WEIGHTS_NAME)):
            raise ValueError("Loading path should be a directory where the head is saved.")

        head_name = None

        # Load head config if available - this is only possible if we have a prediction_heads config
        if isfile(join(save_directory, HEAD_CONFIG_NAME)) and hasattr(self.model.config, 'prediction_heads'):
            config = self.load_weights_config(save_directory)
            head_name = load_as or config['name']
            if head_name in self.model.config.prediction_heads:
                logger.warn("Overwriting existing head '{}'".format(head_name))
            self.model._add_prediction_head(head_name, config['config'], overwrite_ok=True)

        # Load head weights
        filter_func = self._filter_func(head_name)
        if head_name:
            rename_func = self._rename_func(config['name'], head_name)
        else:
            rename_func = None
        self.load_weights(save_directory, filter_func, rename_func=rename_func)

        return head_name


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
            config (str or dict, optional): The adapter configuration, can be either:
                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        pass

    @abstractmethod
    def train_adapter(self, adapter_type: AdapterType):
        """Sets the model in mode for training the given type of adapter.
        """
        pass

    def train_language_adapter(self):
        """Sets the model in mode for training language adapters.
        """
        self.train_adapter(AdapterType.text_lang)

    def train_task_adapter(self):
        """Sets the model in mode for training task adapters.
        """
        self.train_adapter(AdapterType.text_task)

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

    def save_adapter(self, save_directory: str, adapter_name: str, meta_dict=None):
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
        else:
            raise ValueError("Could not resolve '{}' to a valid adapter name.".format(adapter_name))

    def load_adapter(self, adapter_name_or_path, adapter_type=None, config=None,
                     version=None, model_name=None, load_as=None, **kwargs):
        """Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            adapter_type (AdapterType, optional): The type of adapter to be loaded. If not specified, text_task will be used for adapters loaded from the Hub.
            config (str, optional): The requested configuration of the adapter. If not specified, will be either:
                - the default adapter config for the requested adapter if specified
                - the global default adapter config
            version (int, optional): The version of the adapter to be loaded.
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was saved will be used.
        """
        if AdapterType.has(adapter_type) or not adapter_type:
            loader = AdapterLoader(self, adapter_type)
            return loader.load(
                adapter_name_or_path, config, version, model_name, load_as, **kwargs
            )
        else:
            raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def save_all_adapters(self, save_directory: str, meta_dict=None):
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
                meta_dict.update({'config_id': h})
            else:
                meta_dict = {'config_id': h}
            self.save_adapter(save_path, name, meta_dict=meta_dict)

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model.
        """
        # first freeze/ unfreeze all model weights
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
        self.model_freezed = freeze


class ModelWithHeadsAdaptersMixin(ModelAdaptersMixin):
    """Mixin adding support for loading/ saving adapters to transformer models with head(s)."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model_name = None

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

    def train_adapter(self, adapter_type: AdapterType):
        """Sets the model in mode for training the given type of adapter.
        """
        self.base_model.train_adapter(adapter_type)

    def save_head(self, save_directory: str, head_name: str = None):
        loader = PredictionHeadLoader(self)
        loader.save(save_directory, head_name=head_name)

    def load_head(self, save_directory, load_as=None):
        loader = PredictionHeadLoader(self)
        return loader.load(save_directory, load_as=load_as)
