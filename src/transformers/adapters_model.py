import logging
from os import mkdir
from os.path import isdir, isfile, join, exists
import torch
import json
from transformers.file_utils import is_remote_url, get_from_cache, torch_cache_home
from .adapters_utils import find_matching_config_path, urljoin, CONFIG_NAME, WEIGHTS_NAME, HEAD_WEIGHTS_NAME


logger = logging.getLogger(__name__)

# the download cache
ADAPTER_CACHE = join(torch_cache_home, "adapters")

PRETRAINED_TASK_ADAPTER_MAP = {
    "csqa": "http://adapter-hub.webredirect.org/task-csqa/",
    "hellaswag": "http://adapter-hub.webredirect.org/task-hellaswag/",
    "sst": "http://adapter-hub.webredirect.org/task-sst/",
    "multinli": "http://adapter-hub.webredirect.org/task-multinli/",
    "dummy": "http://adapter-hub.webredirect.org/task-dummy/"
}

PRETRAINED_LANG_ADAPTER_MAP = {
    "xyz": "http://adapter-hub.webredirect.org/lang-xyz/"
}

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
        'residual_before_ln': True
    }
}

DEFAULT_ADAPTER_CONFIG = 'pfeiffer'


class ModelAdaptersMixin:
    """Mixin for transformer models adding support for loading/ saving adapters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_adapter_config(self, adapter_config):
        """Sets the adapter configuration of the task adapters.

        Args:
            adapter_config (str or dict): adapter configuration, can be either:
                - a string identifying a pre-defined adapter configuration
                - a dictionary representing the adapter configuration
        """
        if hasattr(self.config, "adapters"):
            assert len(self.config.adapters) < 1, "Can only set new config if no adapters have been added."

        if isinstance(adapter_config, dict):
            self.config.adapter_config = adapter_config
        elif adapter_config in ADAPTER_CONFIG_MAP:
            self.config.adapter_config = ADAPTER_CONFIG_MAP[adapter_config]
        else:
            raise ValueError("Unable to identify {} as a valid adapter config.".format(adapter_config))
        self.config.adapters = []

    def set_language_adapter_config(self, adapter_config):
        """Sets the adapter configuration of the language adapters.

        Args:
            config (str or dict): adapter configuration, can be either:
                - a string identifying a pre-defined adapter configuration
                - a dictionary representing the adapter configuration
        """
        if hasattr(self.config, "language_adapters"):
            assert len(self.config.language_adapters) < 1, "Can only set new config if no adapters have been added."

        if isinstance(adapter_config, dict):
            self.config.adapter_config = adapter_config
        elif adapter_config in ADAPTER_CONFIG_MAP:
            self.config.language_adapter_config = ADAPTER_CONFIG_MAP[adapter_config]
        else:
            raise ValueError("Unable to identify {} as a valid adapter config.".format(adapter_config))
        self.config.language_adapters = []

    def adapter_state_dict(self, adapter_type, adapter_name):
        """Returns a dictionary containing the whole state of the specified task adapter.

        Args:
            task_name (str): the name of the task adapter
        """
        is_part = self._get_params_check_func(adapter_type, adapter_name)
        return {k: v for (k, v) in self.state_dict().items() if is_part(k)}

    # TODO temporary workaround. replace with a better solution after changeing module item names.
    def _get_params_check_func(self, adapter_type, adapter_name):
        if adapter_type == 'task':
            return lambda x: 'adapters.{}'.format(adapter_name) in x and 'language' not in x
        elif adapter_type == 'lang':
            return lambda x: 'adapters.{}'.format(adapter_name) in x and 'language' in x
        elif adapter_type == 'head':
            return lambda x: 'prediction_heads.{}'.format(adapter_name) in x
        else:
            raise ValueError("Invalid adapter type {}".format(adapter_type))

    def adapter_save_config(self, adapter_type, name=None, default_config=None, with_head=False):
        config_dict = {
            'type': adapter_type,
            'model': self.__class__.__name__, 'hidden_size': self.config.hidden_size
        }
        if name:
            config_dict['name'] = name
        if adapter_type == 'task':
            config_dict['config'] = getattr(self.config, "adapter_config", default_config)
        elif adapter_type == 'lang':
            config_dict['config'] = getattr(self.config, "language_adapter_config", default_config)
        else:
            raise ValueError("Invalid adapter type {}".format(adapter_type))
        if with_head:
            config_dict['prediction_head'] = self.config.prediction_heads[name]
        return config_dict

    def save_adapter(self, save_directory, task_name, save_head=False):
        """Saves a task adapter and its configuration file to a directory, so that it can be reloaded
        using the `model.load_adapter()` method.

        Args:
            save_directory (str): a path to a directory where the adapter will be saved
            task_name (str): the name of the task adapter to be saved
        """
        assert task_name in self.config.adapters, "No task adapter with the given name is part of this model."

        config_dict = self.adapter_save_config('task', task_name, with_head=save_head)
        self._save_adapter(save_directory, config_dict, save_head)

    def save_language_adapter(self, save_directory, language_name, save_head=False):
        """Saves a language adapter and its configuration file to a directory, so that it can be reloaded
        using the `model.load_adapter()` method.

        Args:
            save_directory (str): a path to a directory where the adapter will be saved
            task_name (str): the name of the language adapter to be saved
        """
        assert language_name in self.config.language_adapters, "No language adapter with the given name is part of this model."

        config_dict = self.adapter_save_config('lang', language_name, with_head=save_head)
        self._save_adapter(save_directory, config_dict, save_head)

    def _save_adapter(self, save_directory, config_dict, save_head):
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), "Saving path should be a directory where adapter and configuration can be saved."

        # Save the adapter configuration
        output_config_file = join(save_directory, CONFIG_NAME)
        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
        logger.info("Configuration saved in {}".format(output_config_file))

        # Get the state of all adapter modules for this task
        adapter_state_dict = self.adapter_state_dict(config_dict['type'], config_dict['name'])
        # Save the adapter weights
        output_file = join(save_directory, WEIGHTS_NAME)
        torch.save(adapter_state_dict, output_file)
        logger.info("Adapter weights saved in {}".format(output_file))

        # optionally save the prediction head in a separate file
        if save_head:
            heads_state_dict = self.adapter_state_dict('head', config_dict['name'])
            output_file = join(save_directory, HEAD_WEIGHTS_NAME)
            torch.save(heads_state_dict, output_file)
            logger.info("Prediction head weights saved in {}".format(output_file))

    def load_adapter(self, adapter_name_or_path, default_config=DEFAULT_ADAPTER_CONFIG,
                     version=None, load_head=False, **kwargs):
        """Loads a pre-trained pytorch task adapter from an adapter configuration.

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
        resolved_folder = self._resolve_adapter_path(adapter_name_or_path, 'task', default_config, version)

        # Load config of adapter
        config = self._load_adapter_config(resolved_folder, cache_dir=cache_dir, **kwargs)
        assert config['type'] == 'task', "Loaded adapter has to be a task adapter."
        # If no adapter config is available yet, set to the config of the loaded adapter
        if not hasattr(self.config, "adapter_config"):
            self.set_adapter_config(config['config'])
        # Otherwise, check that loaded config is equal to the config of this model.
        else:
            for k, v in config['config'].items():
                assert self.config.adapter_config[k] == v, "Adapter configurations have to be equal."

        # If the adapter is not part of the model, add it
        if config['name'] not in self.config.adapters:
            self.add_adapter(config['name'])

        self._load_adapter_weights(resolved_folder, config['type'], config['name'], cache_dir=cache_dir, **kwargs)

        # Optionally, load the weights of the prediction head
        if load_head:
            self._load_adapter_head(resolved_folder, config, cache_dir=cache_dir, **kwargs)

    def load_language_adapter(self, adapter_name_or_path, default_config=DEFAULT_ADAPTER_CONFIG,
                              version=None, load_head=False, **kwargs):
        """Loads a pre-trained pytorch language adapter from an adapter configuration.

        Args:
            adapter_name_or_path (str): can be either:
                - the identifier of a pre-trained language adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
            default_config (str, optional): The identifier of the adapter configuration to be used if no config is set.
            version (int, optional): The version of the adapter to be loaded.
            load_head (bool, optional): If set to true, load the corresponding prediction head toegether with the adapter. Defaults to False.
        """
        cache_dir = kwargs.pop("cache_dir", ADAPTER_CACHE)

        # Resolve the weights to be loaded based on the given identifier and the current adapter config
        resolved_folder = self._resolve_adapter_path(adapter_name_or_path, 'lang', default_config, version)

        # Load config of adapter
        config = self._load_adapter_config(resolved_folder, cache_dir=cache_dir, **kwargs)
        assert config['type'] == 'lang', "Loaded adapter has to be a language adapter."
        # If no adapter config is available yet, set to the config of the loaded adapter
        if not hasattr(self.config, "language_adapter_config"):
            self.set_language_adapter_config(config['config'])
        # Otherwise, check that loaded config is equal to the config of this model.
        else:
            for k, v in config['config'].items():
                assert self.config.language_adapter_config[k] == v, "Adapter configurations have to be equal."

        # If the adapter is not part of the model, add it
        if config['name'] not in self.config.language_adapters:
            self.add_language_adapter(config['name'])

        self._load_adapter_weights(resolved_folder, config['type'], config['name'], cache_dir=cache_dir, **kwargs)

        # Optionally, load the weights of the prediction head
        if load_head:
            self._load_adapter_head(resolved_folder, config, cache_dir=cache_dir, **kwargs)

    def _load_adapter_head(self, resolved_folder, config, **kwargs):
        """Loads a prediction head for an adapter.
        """
        assert 'prediction_head' in config, "Loaded adapter has no prediction head included."
        head_config = config['prediction_head']

        # Add the prediction head to the model
        # TODO check the case when prediction head is already present
        self.add_prediction_head(
            config['name'], nr_labels=head_config['nr_labels'],
            task_type=head_config['task_type'], layers=head_config['layers'],
            activation_function=head_config['activation_function'], qa_examples=head_config['qa_examples']
        )

        return self._load_adapter_weights(
            resolved_folder, 'head', config['name'], weights_name=HEAD_WEIGHTS_NAME, **kwargs
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

    def _load_adapter_weights(self, resolved_folder, adapter_type, adapter_name, weights_name=WEIGHTS_NAME, **kwargs):
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

        # Add the weights to the model
        missing_keys, unexpected_keys = self.load_state_dict(adapter_state_dict, strict=False)

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

    def _resolve_adapter_path(self, adapter_name_or_path, adapter_type, default_config, version):
        assert default_config in ADAPTER_CONFIG_MAP, "Specified default config is invalid."
        config = self.adapter_save_config('task', default_config=ADAPTER_CONFIG_MAP[default_config])
        # task adapter with identifier
        if adapter_type == 'task' and adapter_name_or_path in PRETRAINED_TASK_ADAPTER_MAP:
            return find_matching_config_path(
                PRETRAINED_TASK_ADAPTER_MAP[adapter_name_or_path], config, version
            )
        # language adapter with identifier
        elif adapter_type == 'lang' and adapter_name_or_path in PRETRAINED_LANG_ADAPTER_MAP:
            return find_matching_config_path(
                PRETRAINED_LANG_ADAPTER_MAP[adapter_name_or_path], config, version
            )
        # url of a folder containing pretrained adapters
        elif is_remote_url(adapter_name_or_path):
            return find_matching_config_path(adapter_name_or_path, config, version)
        # path to a local folder saved using save_adapter() or save_language_adapter()
        elif isdir(adapter_name_or_path):
            if isfile(join(adapter_name_or_path, WEIGHTS_NAME)) and isfile(join(adapter_name_or_path, CONFIG_NAME)):
                return adapter_name_or_path
            else:
                raise EnvironmentError(
                    "No file {} or no file {} found in directory {}".format(
                        WEIGHTS_NAME, CONFIG_NAME, adapter_name_or_path)
                )
        else:
            raise ValueError("Unable to identify {} as a valid module location.".format(adapter_name_or_path))
