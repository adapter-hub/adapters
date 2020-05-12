import logging
from os import mkdir
from os.path import isdir, isfile, join, exists
import torch
import json
import re
from transformers.file_utils import is_remote_url, get_from_cache, torch_cache_home
from .adapters_utils import (
    find_matching_config_path,
    urljoin,
    CONFIG_NAME, WEIGHTS_NAME, HEAD_WEIGHTS_NAME,
    ADAPTER_IDENTIFIER_PATTERN
)
from .adapters_config import (
    ADAPTER_CONFIG_MAP,
    DEFAULT_ADAPTER_CONFIG,
    AdapterType,
    AdapterConfig
)


logger = logging.getLogger(__name__)

# the download cache
ADAPTER_CACHE = join(torch_cache_home, "adapters")

ADAPTER_HUB_URL = "http://adapter-hub.webredirect.org/repo/"


class AdapterLoader:
    """A class providing methods for saving and loading adapter modules of a specified type."""

    def __init__(self, model, adapter_type: AdapterType):
        self.model = model
        self.adapter_type = adapter_type

    # TODO remove this
    @property
    def config(self):
        return self.model.config.adapter_config.get_default(self.adapter_type)

    @config.setter
    def config(self, value):
        self.model.config.adapter_config.set_default(self.adapter_type, value)

    def state_dict(self, adapter_name: str):
        """Returns a dictionary containing the whole state of the specified adapter.

        Args:
            adapter_name (str): the name of the task adapter
        """
        is_part = self._get_params_check_func(self.adapter_type, adapter_name)
        return {k: v for (k, v) in self.model.state_dict().items() if is_part(k)}

    def _get_params_check_func(self, adapter_type, adapter_name):
        if adapter_type == 'head':
            return lambda x: 'prediction_heads.{}'.format(adapter_name) in x
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

    def full_config(self, name=None, default_config=None, with_head=False):
        config_dict = {
            'type': self.adapter_type,
            'model': self.model.__class__.__name__, 'hidden_size': self.model.config.hidden_size
        }
        if name:
            config_dict['name'] = name
        config_dict['config'] = self.config or default_config
        if with_head:
            config_dict['prediction_head'] = self.model.config.prediction_heads[name]
        return config_dict

    def save(self, save_directory, name, save_head=False):
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
        assert name in self.model.config.adapter_config.adapters, "No adapter of this type with the given name is part of this model."

        config_dict = self.full_config(name, with_head=save_head)

        # Save the adapter configuration
        output_config_file = join(save_directory, CONFIG_NAME)
        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
        logger.info("Configuration saved in {}".format(output_config_file))

        # Get the state of all adapter modules for this task
        adapter_state_dict = self.state_dict(config_dict['name'])
        # Save the adapter weights
        output_file = join(save_directory, WEIGHTS_NAME)
        torch.save(adapter_state_dict, output_file)
        logger.info("Adapter weights saved in {}".format(output_file))

    def load(self, adapter_name_or_path, default_config=DEFAULT_ADAPTER_CONFIG,
             version=None, load_head=False, load_as=None, **kwargs):
        """Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
            default_config (str, optional): The identifier of the adapter configuration to be used if no config is set.
            version (int, optional): The version of the adapter to be loaded.
            load_head (bool, optional): If set to true, load the corresponding prediction head toegether with the adapter. Defaults to False.
        """
        cache_dir = kwargs.pop("cache_dir", ADAPTER_CACHE)

        # Resolve the weights to be loaded based on the given identifier and the current adapter config
        resolved_folder = self._resolve_adapter_path(adapter_name_or_path, default_config, version)

        # Load config of adapter
        config = self._load_adapter_config(resolved_folder, cache_dir=cache_dir, **kwargs)
        assert config['type'] == self.adapter_type, "Loaded adapter has to be a {} adapter.".format(self.adapter_type)
        # If no adapter config is available yet, set to the config of the loaded adapter
        if not self.config:
            self.config = config['config']
        # Otherwise, check that loaded config is equal to the config of this model.
        else:
            for k, v in config['config'].items():
                assert self.config[k] == v, "Adapter configurations have to be equal."

        adapter_name = load_as or config['name']
        # If the adapter is not part of the model, add it
        if adapter_name not in self.model.config.adapter_config.adapters:
            self.model.add_adapter(self.adapter_type, adapter_name)

        self._load_adapter_weights(resolved_folder, self.adapter_type, config['name'], load_as=load_as, cache_dir=cache_dir, **kwargs)

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
                resolved_folder, 'head', config['name'], weights_name=HEAD_WEIGHTS_NAME, load_as=load_as, **kwargs
            )

    def _load_adapter_config(self, resolved_folder, **kwargs):
        """Loads an adapter configuration.
        """
        # If necessary, download the config or load it from cache
        if is_remote_url(resolved_folder):
            resolved_file = urljoin(resolved_folder, CONFIG_NAME)
            config_file = get_from_cache(resolved_file, **kwargs)
            if not config_file:
                raise EnvironmentError(
                    "Unable to load file from {}. The file might be unavailable.".format(resolved_file)
                )
            logger.info("loading configuration file {} from cache at {}".format(resolved_file, config_file))
        else:
            config_file = join(resolved_folder, CONFIG_NAME)
            logger.info("loading configuration file {}".format(config_file))

        # Load the config
        with open(config_file, 'r', encoding='utf-8') as f:
            adapter_config = json.load(f)

        return adapter_config

    def _load_adapter_weights(self, resolved_folder, adapter_type, adapter_name, weights_name=WEIGHTS_NAME, load_as=None, **kwargs):
        """Loads adapter weights.
        """
        # If necessary, download the weights or load them from cache
        if is_remote_url(resolved_folder):
            resolved_file = urljoin(resolved_folder, weights_name)
            weights_file = get_from_cache(resolved_file, **kwargs)
            if not weights_file:
                raise EnvironmentError(
                    "Unable to load file from {}. The file might be unavailable.".format(resolved_file)
                )
            logger.info("loading weights file {} from cache at {}".format(resolved_file, weights_file))
        else:
            weights_file = join(resolved_folder, weights_name)
            logger.info("loading weights file {}".format(weights_file))

        # Load the weights of the adapter
        try:
            adapter_state_dict = torch.load(weights_file, map_location="cpu")
        except Exception:
            raise OSError("Unable to load weights from pytorch checkpoint file. ")

        # Rename weights if needed
        if load_as:
            adapter_state_dict = self._rename_params(adapter_state_dict, adapter_name, load_as)
            adapter_name = load_as

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

    def _resolve_adapter_path(self, adapter_name_or_path, default_config, version):
        assert default_config in ADAPTER_CONFIG_MAP, "Specified default config is invalid."
        config = self.full_config(default_config=ADAPTER_CONFIG_MAP[default_config])
        # url of a folder containing pretrained adapters
        if is_remote_url(adapter_name_or_path):
            return find_matching_config_path(adapter_name_or_path, config, version)
        # path to a local folder saved using save()
        elif isdir(adapter_name_or_path):
            if isfile(join(adapter_name_or_path, WEIGHTS_NAME)) and isfile(join(adapter_name_or_path, CONFIG_NAME)):
                return adapter_name_or_path
            else:
                raise EnvironmentError(
                    "No file {} or no file {} found in directory {}".format(
                        WEIGHTS_NAME, CONFIG_NAME, adapter_name_or_path)
                )
        # matches possible form of identifier in hub
        elif re.fullmatch(ADAPTER_IDENTIFIER_PATTERN, adapter_name_or_path):
            url = urljoin(ADAPTER_HUB_URL, self.adapter_type, adapter_name_or_path)
            return find_matching_config_path(url, config, version)
        else:
            raise ValueError("Unable to identify {} as a valid module location.".format(adapter_name_or_path))


class ModelAdaptersMixin:
    """Mixin for transformer models adding support for loading/ saving adapters."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if not hasattr(config, 'adapter_config'):
            config.adapter_config = AdapterConfig()
        self.adapters = {
            t: AdapterLoader(self, t) for t in AdapterType
        }

    def has_adapters(self, adapter_type=None):
        if not adapter_type:
            return len(self.config.adapter_config.adapters) > 0
        else:
            return len(self.config.adapter_config.adapter_list(adapter_type)) > 0

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
            self.config.adapter_config.set_default(adapter_type, adapter_config)
        else:
            raise ValueError("Invalid adapter type {}".format(adapter_type))

    def save_adapter(self, adapter_type, save_directory, adapter_name, save_head=False):
        if AdapterType.has(adapter_type):
            self.adapters[adapter_type].save(save_directory, adapter_name, save_head)
        else:
            raise ValueError("Invalid adapter type {}".format(adapter_type))

    def save_task_adapter(self, save_directory, task_name, save_head=False):
        """Saves a task adapter and its configuration file to a directory, so that it can be reloaded
        using the `model.load_task_adapter()` method.

        Args:
            save_directory (str): a path to a directory where the adapter will be saved
            task_name (str): the name of the task adapter to be saved
        """
        self.adapters[AdapterType.text_task].save(save_directory, task_name, save_head)

    def save_language_adapter(self, save_directory, language_name, save_head=False):
        """Saves a language adapter and its configuration file to a directory, so that it can be reloaded
        using the `model.load_language_adapter()` method.

        Args:
            save_directory (str): a path to a directory where the adapter will be saved
            task_name (str): the name of the language adapter to be saved
        """
        self.adapters[AdapterType.text_lang].save(save_directory, language_name, save_head)

    def load_adapter(self, adapter_type, adapter_name_or_path, default_config=DEFAULT_ADAPTER_CONFIG,
                     version=None, load_head=False, load_as=None, **kwargs):
        if AdapterType.has(adapter_type):
            self.adapters[adapter_type].load(adapter_name_or_path, default_config, version, load_head, load_as, **kwargs)
        else:
            raise ValueError("Invalid adapter type {}".format(adapter_type))

    def load_task_adapter(self, adapter_name_or_path, default_config=DEFAULT_ADAPTER_CONFIG,
                          version=None, load_head=False, load_as=None, **kwargs):
        """Loads a pre-trained pytorch task adapter from an adapter configuration.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
            default_config (str, optional): The identifier of the adapter configuration to be used if no config is set.
            version (int, optional): The version of the adapter to be loaded.
            load_head (bool, optional): If set to true, load the corresponding prediction head toegether with the adapter. Defaults to False.
        """
        self.adapters[AdapterType.text_task].load(adapter_name_or_path, default_config, version, load_head, load_as, **kwargs)

    def load_language_adapter(self, adapter_name_or_path, default_config=DEFAULT_ADAPTER_CONFIG,
                              version=None, load_head=False, load_as=None, **kwargs):
        """Loads a pre-trained pytorch language adapter from an adapter configuration.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained language adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
            default_config (str, optional): The identifier of the adapter configuration to be used if no config is set.
            version (int, optional): The version of the adapter to be loaded.
            load_head (bool, optional): If set to true, load the corresponding prediction head toegether with the adapter. Defaults to False.
        """
        self.adapters[AdapterType.text_lang].load(adapter_name_or_path. default_config, version, load_head, load_as, **kwargs)
