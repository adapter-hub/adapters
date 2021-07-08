import logging
import warnings
from abc import ABC, abstractmethod
from os.path import join
from typing import List, Mapping, Optional, Union

from torch import nn

from .composition import AdapterCompositionBlock, Fuse, Stack, parse_composition
from .configuration import (
    ADAPTERFUSION_CONFIG_MAP,
    DEFAULT_ADAPTERFUSION_CONFIG,
    AdapterConfig,
    AdapterFusionConfig,
    ModelAdaptersConfig,
    get_adapter_config_hash,
)
from .hub_mixin import PushAdapterToHubMixin
from .loading import AdapterFusionLoader, AdapterLoader, PredictionHeadLoader, WeightsLoader
from .modeling import Adapter, GLOWCouplingBlock, NICECouplingBlock
from .utils import inherit_doc


logger = logging.getLogger(__name__)


class InvertibleAdaptersMixin:
    """Mixin for Transformer models adding invertible adapters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invertible_adapters = nn.ModuleDict(dict())

    def add_invertible_adapter(self, adapter_name: str):
        """
        Adds an invertible adapter module for the adapter with the given name. If the given adapter does not specify an
        invertible adapter config, this method does nothing.

        Args:
            adapter_name (str): The name of the adapter for which to add an invertible adapter module.
        """
        if adapter_name in self.invertible_adapters:
            raise ValueError(f"Model already contains an adapter module for '{adapter_name}'.")
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config["inv_adapter"]:
            if adapter_config["inv_adapter"] == "nice":
                inv_adap = NICECouplingBlock(
                    [[self.config.hidden_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            elif adapter_config["inv_adapter"] == "glow":
                inv_adap = GLOWCouplingBlock(
                    [[self.config.hidden_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            else:
                raise ValueError(f"Invalid invertible adapter type '{adapter_config['inv_adapter']}'.")
            self.invertible_adapters[adapter_name] = inv_adap
            self.invertible_adapters[adapter_name].apply(Adapter.init_bert_weights)

    def delete_invertible_adapter(self, adapter_name: str):
        if adapter_name in self.invertible_adapters:
            del self.invertible_adapters[adapter_name]

    def get_invertible_adapter(self):
        # TODO: Currently no fusion over invertible adapters, takes only very first language adapter position
        if self.config.adapters.active_setup is not None and len(self.config.adapters.active_setup) > 0:
            first_adapter = self.config.adapters.active_setup.first()
            if first_adapter in self.invertible_adapters:
                return self.invertible_adapters[first_adapter]
        return None

    def enable_invertible_adapters(self, adapter_names):
        for adapter_name in adapter_names:
            if adapter_name in self.invertible_adapters:
                for param in self.invertible_adapters[adapter_name].parameters():
                    param.requires_grad = True

    def invertible_adapters_forward(self, hidden_states, rev=False):
        # TODO: Currently no fusion over invertible adapters, takes only very first language adapter position
        if self.config.adapters.active_setup is not None and len(self.config.adapters.active_setup) > 0:
            first_adapter = self.config.adapters.active_setup.first()
            if first_adapter in self.invertible_adapters:
                hidden_states = self.invertible_adapters[first_adapter](hidden_states, rev=rev)

        return hidden_states


class ModelConfigAdaptersMixin(ABC):
    """
    Mixin for model config classes, adding support for adapters.

    Besides adding this mixin to the config class of a model supporting adapters, make sure the following attributes/
    properties are present: hidden_dropout_prob, attention_probs_dropout_prob.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # adapter configuration
        adapter_config_dict = kwargs.pop("adapters", None)
        if adapter_config_dict:
            self.adapters = ModelAdaptersConfig(**adapter_config_dict)
        else:
            self.adapters = ModelAdaptersConfig()


class ModelAdaptersMixin(PushAdapterToHubMixin, ABC):
    """Mixin for transformer models adding support for loading/ saving adapters."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model_name = None

        # In some cases, the config is not an instance of a directly supported config class such as BertConfig.
        # Thus, we check the adapters config here to make sure everything is correct.
        if not hasattr(config, "adapters"):
            config.adapters = ModelAdaptersConfig()
        elif not isinstance(config.adapters, ModelAdaptersConfig):
            config.adapters = ModelAdaptersConfig(**config.adapters)

    def _init_adapter_modules(self):
        """
        This method initializes adapter modules and fusion modules from the model config.
        """
        # Initialize adapters from config
        for adapter_name in self.config.adapters:
            self._add_adapter(adapter_name)
        # Initialize fusion from config
        if hasattr(self.config, "adapter_fusion_models"):
            for fusion_adapter_names in self.config.adapter_fusion_models:
                self._add_fusion_layer(fusion_adapter_names)

    # These methods have to be implemented by every deriving class:

    @abstractmethod
    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training the given adapters."""
        pass

    def train_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        warnings.warn(
            "add_fusion() has been deprecated in favor of add_adapter_fusion(). Please use the newer method instead.",
            FutureWarning,
        )
        self.train_adapter_fusion(adapter_setup, unfreeze_adapters=unfreeze_adapters)

    @abstractmethod
    def train_adapter_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        pass

    @abstractmethod
    def _add_adapter(self, adapter_name):
        pass

    @abstractmethod
    def _add_fusion_layer(self, adapter_names):
        pass

    def has_adapters(self):
        return len(self.config.adapters.adapters) > 0

    @property
    def has_parallel_adapters(self) -> bool:
        if self.config.adapters.active_setup:
            return self.config.adapters.active_setup.parallel_channels > 1
        else:
            return False

    @property
    def active_adapters(self) -> AdapterCompositionBlock:
        return self.config.adapters.active_setup

    @active_adapters.setter
    def active_adapters(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        self.set_active_adapters(adapter_setup)

    def set_active_adapters(
        self, adapter_setup: Union[list, AdapterCompositionBlock], skip_layers: Optional[List[int]] = None
    ):
        """
        Sets the adapter modules to be used by default in every forward pass. If no adapter with the given name is
        found, no module of the respective type will be activated.

        Args:
            adapter_setup (list): The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        adapter_setup = parse_composition(adapter_setup, model_type=self.config.model_type)
        if adapter_setup:
            for adapter_name in adapter_setup.flatten():
                if adapter_name not in self.config.adapters.adapters:
                    raise ValueError(
                        f"No adapter with name '{adapter_name}' found. Please make sure that all specified adapters are correctly loaded."
                    )

        self.config.adapters.active_setup = adapter_setup
        self.config.adapters.skip_layers = skip_layers

    def set_adapter_fusion_config(self, adapter_fusion_config, override_kwargs=None):
        """
        Sets the adapter fusion configuration.

        Args:
            adapter_fusion_config (str or dict): adapter fusion configuration, can be either:

                - a string identifying a pre-defined adapter fusion configuration
                - a dictionary representing the adapter fusion configuration
                - the path to a file containing the adapter fusion configuration
        """
        if override_kwargs is None:
            override_kwargs = {}
        if isinstance(adapter_fusion_config, str) and adapter_fusion_config in ADAPTERFUSION_CONFIG_MAP:
            self.config.adapter_fusion = AdapterFusionConfig.load(adapter_fusion_config, **override_kwargs)
        elif isinstance(adapter_fusion_config, Mapping):
            self.config.adapter_fusion = adapter_fusion_config
        else:
            raise ValueError("Invalid adapter type {}".format(adapter_fusion_config))

    def add_adapter(self, adapter_name: str, config=None, overwrite_ok: bool = False):
        """
        Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:

                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
            overwrite_ok (bool, optional): Overwrite an adapter with the same name if it exists. By default (False), an exception is thrown.
        """
        if isinstance(config, dict):
            config = AdapterConfig.from_dict(config)  # ensure config is ok and up-to-date
        # In case adapter already exists and we allow overwriting, explicitly delete the existing one first
        if overwrite_ok and adapter_name in self.config.adapters:
            self.delete_adapter(adapter_name)
        self.config.adapters.add(adapter_name, config=config)
        self.base_model._add_adapter(adapter_name)

    def add_fusion(self, adapter_names: Union[Fuse, list], adapter_fusion_config=None, override_kwargs=None):
        warnings.warn(
            "add_fusion() has been deprecated in favor of add_adapter_fusion(). Please use the newer method instead.",
            FutureWarning,
        )
        self.add_adapter_fusion(adapter_names, adapter_fusion_config, override_kwargs)

    def add_adapter_fusion(self, adapter_names: Union[Fuse, list], adapter_fusion_config=None, override_kwargs=None):
        """
        Adds AdapterFusion to the model with alll the necessary configurations and weight initializations

        Args:
            adapter_names: a list of adapter names which should be fused
            adapter_fusion_config (str or dict): adapter fusion configuration, can be either:

                - a string identifying a pre-defined adapter fusion configuration
                - a dictionary representing the adapter fusion configuration
                - the path to a file containing the adapter fusion configuration
            override_kwargs: dictionary items for values which should be overwritten in the default AdapterFusion configuration
        """
        # TODO-V2 Allow nested items or directly pass Fuse block?
        if isinstance(adapter_names, Fuse):
            adapter_names = adapter_names.children
        if not hasattr(self.config, "adapter_fusion"):
            if override_kwargs is None:
                override_kwargs = {}
            if adapter_fusion_config is not None:
                self.set_adapter_fusion_config(adapter_fusion_config, **override_kwargs)
            else:
                self.set_adapter_fusion_config(DEFAULT_ADAPTERFUSION_CONFIG)
        elif hasattr(self.config, "adapter_fusion") and adapter_fusion_config is not None:
            # This behavior may be a bit unintuitive as the given argument is ignored, but we can't throw an error because of the loader.
            logger.warning("An AdapterFusion config has already been set and will NOT be overwritten")

        if not hasattr(self.config, "adapter_fusion_models"):
            self.config.adapter_fusion_models = []
        if isinstance(adapter_names, list):
            adapter_fusion_name = ",".join(adapter_names)
        else:
            adapter_fusion_name = adapter_names
        if adapter_fusion_name not in self.config.adapter_fusion_models:
            self.config.adapter_fusion_models.append(adapter_fusion_name)
            self.base_model._add_fusion_layer(adapter_names)

    def delete_adapter(self, adapter_name: str):
        """
        Deletes the adapter with the specified name from the model.

        Args:
            adapter_name (str): The name of the adapter.
        """
        if adapter_name not in self.config.adapters:
            logger.info("No adapter '%s' found for deletion. Skipping.", adapter_name)
            return
        del self.config.adapters.adapters[adapter_name]
        self.base_model._delete_adapter(adapter_name)
        # Reset active adapters if this was the only active adapter
        if self.active_adapters == Stack(adapter_name):
            self.active_adapters = None

    def delete_adapter_fusion(self, adapter_names: Union[Fuse, list]):
        """
        Deletes the AdapterFusion layer of the specified adapters.

        Args:
            adapter_names (Union[Fuse, list]): List of adapters for which to delete the AdapterFusion layer.
        """
        if isinstance(adapter_names, Fuse):
            adapter_fusion_name = ",".join(adapter_names.children)
        elif isinstance(adapter_names, list):
            adapter_fusion_name = ",".join(adapter_names)
        else:
            adapter_fusion_name = adapter_names

        if (
            not hasattr(self.config, "adapter_fusion_models")
            or adapter_fusion_name not in self.config.adapter_fusion_models
        ):
            logger.info("No AdapterFusion '%s' found for deletion. Skipping.", adapter_fusion_name)
            return
        self.config.adapter_fusion_models.remove(adapter_fusion_name)
        self.base_model._delete_fusion_layer(adapter_fusion_name)

    def save_adapter(
        self,
        save_directory: str,
        adapter_name: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves an adapter and its configuration file to a directory so that it can be shared or reloaded using
        `load_adapter()`.

        Args:
            save_directory (str): Path to a directory where the adapter should be saved.
            adapter_name (str): Name of the adapter to be saved.

        Raises:
            ValueError: If the given adapter name is invalid.
        """
        loader = AdapterLoader(self)
        loader.save(save_directory, adapter_name, meta_dict)
        # save additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.save(save_directory, adapter_name)

    def save_adapter_fusion(
        self,
        save_directory: str,
        adapter_names: list,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves an adapter and its configuration file to a directory so that it can be shared or reloaded using
        `load_adapter()`.

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
        config: Union[dict, str] = None,
        version: str = None,
        model_name: str = None,
        load_as: str = None,
        source: str = "ah",
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        leave_out: Optional[List[int]] = None,
        id2label=None,
        **kwargs
    ) -> str:
        """
        Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:

                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (dict or str, optional): The requested configuration of the adapter.
                If not specified, will be either: - the default adapter config for the requested adapter if specified -
                the global default adapter config
            version (str, optional): The version of the adapter to be loaded.
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
                    saved will be used.
            source (str, optional): Identifier of the source(s) from where to load the adapter. Can be:

                - "ah" (default): search on AdapterHub.
                - "hf": search on HuggingFace model hub.
                - None: only search on local file system
            leave_out: Dynamically drop adapter modules in the specified Transformer layers when loading the adapter.

        Returns:
            str: The name with which the adapter was added to the model.
        """
        loader = AdapterLoader(self)
        load_dir, load_name = loader.load(
            adapter_name_or_path, config, version, model_name, load_as, source=source, leave_out=leave_out, **kwargs
        )
        # load additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(
                    load_dir,
                    load_as=load_as,
                    loading_info=kwargs.get("loading_info", None),
                    main_load_name=load_name,
                    id2label=id2label,
                )
        return load_name

    def load_adapter_fusion(
        self,
        adapter_fusion_name_or_path: str,
        load_as: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        **kwargs
    ) -> str:
        """
        Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_fusion_name_or_path (str): can be either:

                - the identifier of a pre-trained task adapter fusion module to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (dict or str, optional): The requested configuration of the adapter fusion.
                If not specified, will be either: - the default adapter config for the requested adapter fusion if
                specified - the global default adapter fusion config
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
                weights_loader.load(
                    load_dir,
                    load_as=load_as,
                    loading_info=kwargs.get("loading_info", None),
                    main_load_name=load_name,
                )
        return load_name

    def save_all_adapters(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves all adapters of this model together with their configuration to subfolders of the given location.

        Args:
            save_directory (str): Path to a directory where the adapters should be saved.
        """
        for name in self.config.adapters:
            adapter_config = self.config.adapters.get(name)
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
        """
        Saves all adapters of this model together with their configuration to subfolders of the given location.

        Args:
            save_directory (str): Path to a directory where the adapters should be saved.
        """
        if not hasattr(self.config, "adapter_fusion_models"):
            return
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
        """Freezes all weights of the model."""
        # first freeze/ unfreeze all model weights
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
        self.model_freezed = freeze

    def pre_transformer_forward(self, **kwargs):
        """
        This method should be called by every adapter-implementing model at the very beginning of the forward() method.
        """
        # some warnings if we don't use available adapters
        active_adapters = self.active_adapters or kwargs.get("adapter_names", None)
        if not active_adapters and self.has_adapters():
            logger.warning("There are adapters available but none are activated for the forward pass.")

        self.config.adapters.is_parallelized = False


@inherit_doc
class ModelWithHeadsAdaptersMixin(ModelAdaptersMixin):
    """Mixin adding support for loading/ saving adapters to transformer models with head(s)."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._convert_to_flex_head = False

    def add_adapter(self, adapter_name: str, config=None, overwrite_ok: bool = False):
        """
        Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            config (str or dict, optional): The adapter configuration, can be either:

                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
            overwrite_ok (bool, optional): Overwrite an adapter with the same name if it exists. By default (False), an exception is thrown.
        """
        self.base_model.add_adapter(adapter_name, config, overwrite_ok=overwrite_ok)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training the given adapters."""
        self.base_model.train_adapter(adapter_setup)

    def train_adapter_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.base_model.train_adapter_fusion(adapter_setup, unfreeze_adapters=unfreeze_adapters)

    def _add_adapter(self, adapter_name):
        self.base_model._add_adapter(adapter_name)

    def _add_fusion_layer(self, adapter_names):
        self.base_model._add_fusion_layer(adapter_names)

    def save_head(self, save_directory: str, head_name: str = None):
        loader = PredictionHeadLoader(self)
        loader.save(save_directory, name=head_name)

    def load_head(self, save_directory, load_as=None, id2label=None, **kwargs):
        loader = PredictionHeadLoader(self, convert_to_flex_head=self._convert_to_flex_head)
        return loader.load(save_directory, load_as=load_as, id2label=id2label, **kwargs)

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
            custom_weights_loaders.append(PredictionHeadLoader(self, error_on_missing=False))
        super().save_adapter(
            save_directory,
            adapter_name,
            meta_dict=meta_dict,
            custom_weights_loaders=custom_weights_loaders,
        )

    def load_adapter(
        self,
        adapter_name_or_path: str,
        config: Union[dict, str] = None,
        version: str = None,
        model_name: str = None,
        load_as: str = None,
        source: str = "ah",
        with_head: bool = True,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        leave_out: Optional[List[int]] = None,
        id2label=None,
        **kwargs
    ) -> str:
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            custom_weights_loaders.append(
                PredictionHeadLoader(
                    self,
                    error_on_missing=False,
                    convert_to_flex_head=self._convert_to_flex_head,
                )
            )
        # Support passing a num_labels for compatibility reasons. Convert to label map here.
        num_labels = kwargs.pop("num_labels", None)
        if num_labels is not None:
            id2label = {i: "LABEL_" + str(i) for i in range(num_labels)}
        return super().load_adapter(
            adapter_name_or_path,
            config=config,
            version=version,
            model_name=model_name,
            load_as=load_as,
            source=source,
            custom_weights_loaders=custom_weights_loaders,
            leave_out=leave_out,
            id2label=id2label,
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
            save_directory,
            meta_dict=meta_dict,
            custom_weights_loaders=custom_weights_loaders,
        )

    def get_labels(self):
        return list(self.config.id2label.values())

    def get_labels_dict(self):
        return self.config.id2label

    def get_adapter(self, name):
        return self.base_model.get_adapter(name)
