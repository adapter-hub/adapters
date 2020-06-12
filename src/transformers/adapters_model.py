from abc import ABC, abstractmethod
import logging
from os import mkdir
from os.path import isdir, join, exists
import torch
import json
from .adapters_utils import (
    CONFIG_NAME, WEIGHTS_NAME, HEAD_WEIGHTS_NAME,
    resolve_adapter_config,
    resolve_adapter_path,
)
from .adapters_config import (
    DEFAULT_ADAPTER_CONFIG,
    AdapterType,
    build_full_config,
    get_adapter_config_hash,
)


logger = logging.getLogger(__name__)


class AdapterLoader:
    """A class providing methods for saving and loading adapter modules from the Hub, the filesystem or a remote url."""

    def __init__(self, model, adapter_type=None):
        self.model = model
        self.adapter_type = adapter_type

    @property
    def config(self):
        return self.model.config.adapters.get_config(self.adapter_type)

    def _state_dict(self, adapter_type, adapter_name):
        is_part = self._get_params_check_func(adapter_type, adapter_name)
        return {k: v for (k, v) in self.model.state_dict().items() if is_part(k)}

    def _get_params_check_func(self, adapter_type, adapter_name):
        if adapter_type == 'head':
            return lambda x: 'prediction_heads.{}'.format(adapter_name) in x
        elif adapter_type == AdapterType.text_lang:
            return lambda x: '{}_adapters.{}'.format(adapter_type, adapter_name) in x \
                or 'invertible_lang_adapters.{}'.format(adapter_name) in x
        elif AdapterType.has(adapter_type):
            return lambda x: '{}_adapters.{}'.format(adapter_type, adapter_name) in x
        else:
            raise ValueError("Invalid adapter type {}".format(adapter_type))

    def _rename_params(self, state_dict, old_name, new_name):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('_adapters.{}'.format(old_name), '_adapters.{}'.format(new_name)) \
                     .replace('prediction_heads.{}'.format(old_name), 'prediction_heads.{}'.format(new_name))
            new_state_dict[new_k] = v
        return new_state_dict

    def save(self, save_directory, name, save_head=False, meta_dict=None):
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
            adapter_config, adapter_type, self.model.config,
            model_name=self.model.model_name, name=name, with_head=save_head
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

        # Get the state of all adapter modules for this task
        adapter_state_dict = self._state_dict(adapter_type, config_dict['name'])
        # Save the adapter weights
        output_file = join(save_directory, WEIGHTS_NAME)
        torch.save(adapter_state_dict, output_file)
        logger.info("Adapter weights saved in {}".format(output_file))

    def load(self, adapter_name_or_path, config=None,
             version=None, model_name=None, load_head=False, load_as=None, **kwargs):
        """Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (str, optional): The requested configuration of the adapter.
            version (int, optional): The version of the adapter to be loaded.
            model_name (str, optional): The string identifier of the pre-trained model.
            load_head (bool, optional): If set to true, load the corresponding prediction head toegether with the adapter. Defaults to False.
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
        config = self._load_adapter_config(resolved_folder)
        if self.adapter_type:
            assert config['type'] == self.adapter_type, "Loaded adapter has to be a {} adapter.".format(self.adapter_type)

        adapter_name = load_as or config['name']
        # If the adapter is not part of the model, add it
        if adapter_name not in self.model.config.adapters.adapters:
            self.model.add_adapter(adapter_name, config['type'], config=config['config'])
        else:
            logger.warn("Overwriting existing adapter '{}'.".format(adapter_name))

        self._load_adapter_weights(resolved_folder, config['type'], config['name'], load_as=load_as)

        # Optionally, load the weights of the prediction head
        if load_head:
            assert 'prediction_head' in config, "Loaded adapter has no prediction head included."
            head_config = config['prediction_head']

            # Add the prediction head to the model
            # TODO check the case when prediction head is already present
            self.model.add_prediction_head(
                adapter_name, nr_labels=head_config['nr_labels'],
                task_type=head_config['task_type'], layers=head_config['layers'],
                activation_function=head_config['activation_function'], qa_examples=head_config['qa_examples']
            )
            # Load the head weights
            self._load_adapter_weights(
                resolved_folder, 'head', config['name'], weights_name=HEAD_WEIGHTS_NAME, load_as=load_as
            )
        return adapter_name

    def _load_adapter_config(self, resolved_folder):
        """Loads an adapter configuration.
        """
        config_file = join(resolved_folder, CONFIG_NAME)
        logger.info("loading adapter configuration from {}".format(config_file))

        # Load the config
        with open(config_file, 'r', encoding='utf-8') as f:
            adapter_config = json.load(f)

        return adapter_config

    def _load_adapter_weights(self, resolved_folder, adapter_type, adapter_name, weights_name=WEIGHTS_NAME, load_as=None):
        """Loads adapter weights.
        """
        weights_file = join(resolved_folder, weights_name)

        # Load the weights of the adapter
        try:
            adapter_state_dict = torch.load(weights_file, map_location="cpu")
        except Exception:
            raise OSError("Unable to load weights from pytorch checkpoint file. ")

        # Rename weights if needed
        if load_as:
            adapter_state_dict = self._rename_params(adapter_state_dict, adapter_name, load_as)
            adapter_name = load_as
        logger.info("loading adapter weights from {} as '{}'".format(weights_file, adapter_name))

        # Add the weights to the model
        missing_keys, unexpected_keys = self.model.load_state_dict(adapter_state_dict, strict=False)

        params_check = self._get_params_check_func(adapter_type, adapter_name)
        missing_adapter_keys = [k for k in missing_keys if params_check(k)]
        if len(missing_adapter_keys) > 0:
            logger.warn(
                "Some adapter weights could not be found in loaded weights file: {}".format(
                    ', '.join(missing_adapter_keys))
            )
        if len(unexpected_keys) > 0:
            logger.warn(
                "Some weights of the adapter state_dict could not be loaded into model: {}".format(
                    ', '.join(unexpected_keys))
            )

        return missing_adapter_keys, unexpected_keys


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

    def save_adapter(self, save_directory: str, adapter_name: str, save_head=False, meta_dict=None):
        """Saves an adapter and its configuration file to a directory so that it can be shared
        or reloaded using `load_adapter()`.

        Args:
            save_directory (str): Path to a directory where the adapter should be saved.
            adapter_name (str): Name of the adapter to be saved.
            save_head (bool, optional): If set to true, save the matching prediction head for this adapter. Defaults to False.

        Raises:
            ValueError: If the given adapter name is invalid.
        """
        adapter_type = self.config.adapters.get_type(adapter_name)
        if adapter_type:
            loader = AdapterLoader(self, adapter_type)
            loader.save(save_directory, adapter_name, save_head, meta_dict)
        else:
            raise ValueError("Could not resolve '{}' to a valid adapter name.".format(adapter_name))

    def load_adapter(self, adapter_name_or_path, adapter_type=None, config=None,
                     version=None, model_name=None, load_head=False, load_as=None, **kwargs):
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
            load_head (bool, optional): If set to true, load the corresponding prediction head toegether with the adapter. Defaults to False.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was saved will be used.
        """
        if AdapterType.has(adapter_type) or not adapter_type:
            loader = AdapterLoader(self, adapter_type)
            return loader.load(
                adapter_name_or_path, config, version, model_name, load_head, load_as, **kwargs
            )
        else:
            raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def save_all_adapters(self, save_directory: str, save_head=False, meta_dict=None):
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
            self.save_adapter(save_path, name, save_head=save_head, meta_dict=meta_dict)

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model.
        """
        # first freeze/ unfreeze all model weights
        for param in self.parameters():
            param.requires_grad = not freeze
        self.model_freezed = freeze
