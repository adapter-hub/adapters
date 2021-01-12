import torch
from torch import nn

from .adapter_bert import (
    BertEncoderAdaptersMixin,
    BertModelHeadsMixin,
    BertOutputAdaptersMixin,
    BertSelfOutputAdaptersMixin,
)
from .adapter_model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin
from .adapter_utils import AdapterType, flatten_adapter_names


class DistilBertSelfAttentionAdaptersModule(nn.Module, BertSelfOutputAdaptersMixin):
    """Adds attention adapters to the Transformer module of DistilBert."""

    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def layer_norm(self):
        return self.parent.sa_layer_norm


class DistilBertOutputAdaptersModule(nn.Module, BertOutputAdaptersMixin):
    """Adds output adapters to the Transformer module of DistilBert."""

    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def layer_norm(self):
        return self.parent.output_layer_norm


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

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        self.attention_adapters.add_adapter(adapter_name, adapter_type)
        self.output_adapters.add_adapter(adapter_name, adapter_type)

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

    def train_adapter(self, adapter_names: list):
        """Sets the model in mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        adapter_names_flat = flatten_adapter_names(adapter_names)
        self.transformer.enable_adapters(adapter_names, True, False)
        self.enable_invertible_adapters(adapter_names_flat)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_names)

    def train_fusion(self, adapter_names: list):
        """Sets the model in mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_names_flat = flatten_adapter_names(adapter_names)
        self.transformer.enable_adapters(adapter_names_flat, False, True)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_names)

    def _add_adapter(self, adapter_name, adapter_type):
        self.transformer.add_adapter(adapter_name, adapter_type)
        if adapter_type == AdapterType.text_lang:
            self.add_invertible_lang_adapter(adapter_name)

    def _add_fusion_layer(self, adapter_names):
        self.transformer.add_fusion_layer(adapter_names)

    def get_fusion_regularization_loss(self):
        reg_loss = 0.0
        target = torch.zeros((self.config.hidden_size, self.config.hidden_size)).fill_diagonal_(1.0).to(self.device)
        for _, v in self.transformer.layer._modules.items():

            for _, layer_fusion in v.output_adapters.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

            for _, layer_fusion in v.attention_adapters.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

        return reg_loss


class DistilBertModelHeadsMixin(BertModelHeadsMixin):
    """Adds heads to a DistilBert model."""

    pass
