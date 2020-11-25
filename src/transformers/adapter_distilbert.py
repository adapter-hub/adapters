from typing import Union

from torch import nn

from .adapter_bert import (
    BertEncoderAdaptersMixin,
    BertModelHeadsMixin,
    BertOutputAdaptersMixin,
    BertSelfOutputAdaptersMixin,
)
from .adapter_composition import AdapterCompositionBlock, parse_composition
from .adapter_config import DEFAULT_ADAPTER_CONFIG
from .adapter_model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


class DistilBertSelfAttentionAdaptersModule(nn.Module, BertSelfOutputAdaptersMixin):
    """Adds attention adapters to the Transformer module of DistilBert."""

    def __init__(self, parent):
        super().__init__()
        self._layer_norm = parent.sa_layer_norm
        self.config = parent.config

    @property
    def layer_norm(self):
        return self._layer_norm


class DistilBertOutputAdaptersModule(nn.Module, BertOutputAdaptersMixin):
    """Adds output adapters to the Transformer module of DistilBert."""

    def __init__(self, parent):
        super().__init__()
        self._layer_norm = parent.output_layer_norm
        self.config = parent.config

    @property
    def layer_norm(self):
        return self._layer_norm


class DistilBertTransfomerBlockAdaptersMixin:
    """Adds adapters to the TransformerBlock module of DistilBert."""

    def _init_adapter_modules(self):
        self.attention_adapters = DistilBertSelfAttentionAdaptersModule(self)
        self.output_adapters = DistilBertOutputAdaptersModule(self)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()

    def add_fusion_layer(self, adapter_names):
        self.attention_adapters.add_fusion_layer(adapter_names)
        self.output_adapters.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str):
        self.attention_adapters.add_adapter(adapter_name)
        self.output_adapters.add_adapter(adapter_name)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        self.attention_adapters.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)
        self.output_adapters.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)


class DistilBertTransformerAdaptersMixin(BertEncoderAdaptersMixin):
    """Adds adapters to the Transformer module of DistilBert."""

    pass


class DistilBertModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the DistilBert module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        super()._init_adapter_modules()

        # add adapters specified in config; invertible adapter will only be added if required
        for adapter_name in self.config.adapters.adapters:
            self.encoder.add_adapter(adapter_name)
            self.add_invertible_adapter(adapter_name)
        # fusion
        if hasattr(self.config, "fusion_models"):
            for fusion_adapter_names in self.config.fusion_models:
                self.add_fusion_layer(fusion_adapter_names)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.encoder.enable_adapters(adapter_setup.flatten(), True, False)
        self.enable_invertible_adapters(adapter_setup.flatten())
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)

    def train_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.encoder.enable_adapters(adapter_setup.flatten(), False, True)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)

    def add_adapter(self, adapter_name: str, config=None):
        """
        Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:

                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        self.config.adapters.add(adapter_name, config=config)
        self.encoder.add_adapter(adapter_name)
        self.add_invertible_adapter(adapter_name)

    def _add_fusion_layer(self, adapter_names):
        self.transformer.add_fusion_layer(adapter_names)


class DistilBertModelHeadsMixin(BertModelHeadsMixin):
    """Adds heads to a DistilBert model."""

    pass
