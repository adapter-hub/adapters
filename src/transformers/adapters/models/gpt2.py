from typing import Union

import torch
from torch import nn

from ..composition import AdapterCompositionBlock, parse_composition
from ..heads import CausalLMHead, ClassificationHead, MultiLabelClassificationHead, TaggingHead
from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin
from .bert import (
    BertEncoderAdaptersMixin,
    BertOutputAdaptersMixin,
    BertSelfOutputAdaptersMixin,
    ModelWithFlexibleHeadsAdaptersMixin,
)


class GPT2AttentionAdaptersModule(BertSelfOutputAdaptersMixin, nn.Module):
    """Adds attention adapters to the Transformer module of DistilBert."""

    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def transformer_layer_norm(self):
        return None


class GPT2OutputAdaptersModule(BertOutputAdaptersMixin, nn.Module):
    """Adds output adapters to the Transformer module of DistilBert."""

    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def transformer_layer_norm(self):
        return None


class GPT2DecoderBlockAdaptersMixin(BertEncoderAdaptersMixin):
    """Adds adapters to the TransformerBlock module of DistilBert."""

    def _init_adapter_modules(self):
        self.attention_adapters = GPT2AttentionAdaptersModule(self)
        self.output_adapters = GPT2OutputAdaptersModule(self)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()
        self.register_forward_pre_hook(self._adapter_block_pre_hook)

    def add_fusion_layer(self, adapter_names):
        self.attention_adapters.add_fusion_layer(adapter_names)
        self.output_adapters.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.attention_adapters.add_adapter(adapter_name, layer_idx)
        self.output_adapters.add_adapter(adapter_name, layer_idx)

    def delete_adapter(self, adapter_name):
        self.attention_adapters.delete_adapter(adapter_name)
        self.output_adapters.delete_adapter(adapter_name)

    def delete_fusion_layer(self, adapter_names):
        self.attention_adapters.delete_fusion_layer(adapter_names)
        self.output_adapters.delete_fusion_layer(adapter_names)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        self.attention_adapters.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)
        self.output_adapters.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)

    # Makes sure the "parent" reference always points to the correct module.
    # This is especially relevant when using torch data parallelism.
    @staticmethod
    def _adapter_block_pre_hook(module, input_tensors):
        object.__setattr__(module.attention_adapters, "parent", module)
        object.__setattr__(module.output_adapters, "parent", module)


class GPT2ModelAdapterMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        super()._init_adapter_modules()

        # add adapters specified in config; invertible adapter will only be added if required
        for adapter_name in self.config.adapters.adapters:
            self._add_adapter(adapter_name)
        # fusion
        if hasattr(self.config, "fusion_models"):
            for fusion_adapter_names in self.config.fusion_models:
                self.add_fusion_layer(fusion_adapter_names)

    def _add_adapter(self, adapter_name: str):
        adapter_config = self.config.adapters.get(adapter_name)
        leave_out = adapter_config.get("leave_out", [])
        for i, layer in enumerate(self.base_model.h):
            if i not in leave_out:
                layer.add_adapter(adapter_name, i)

        self.add_invertible_adapter(adapter_name)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock], train_embeddings=False):
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.enable_adapters(adapter_setup, True, False)
        self.enable_invertible_adapters(adapter_setup.flatten())
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)
        if train_embeddings:
            self.get_input_embeddings().train()

    def train_adapter_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.enable_adapters(adapter_setup, unfreeze_adapters, True)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)

    def enable_adapters(
        self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool
    ):
        for layer in self.base_model.h:
            layer.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)

    def adjust_attention_mask_for_parallel(self, hidden_states, attention_mask):
        if attention_mask is not None and hidden_states.shape[0] != attention_mask.shape[0]:
            repeats = [1] * len(attention_mask.shape)
            repeats[0] = hidden_states.shape[0] // attention_mask.shape[0]
            attention_mask = attention_mask.repeat(*repeats)
        return attention_mask

    def _add_fusion_layer(self, adapter_names):
        for layer in self.base_model.h:
            layer.add_fusion_layer(adapter_names)

    def _delete_adapter(self, adapter_name: str):
        for layer in self.base_model.h:
            layer.delete_adapter(adapter_name)
        self.delete_invertible_adapter(adapter_name)

    def _delete_fusion_layer(self, adapter_names):
        for layer in self.base_model.h:
            layer.delete_fusion_layer(adapter_names)

    def get_fusion_regularization_loss(self):
        reg_loss = 0.0
        target = torch.zeros((self.config.hidden_size, self.config.hidden_size)).fill_diagonal_(1.0).to(self.device)
        for _, v in self.base_model.h._modules.items():

            for _, layer_fusion in v.output_adapters.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

            for _, layer_fusion in v.attention_adapters.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

        return reg_loss

    def get_adapter(self, name):
        return_adapters = {}
        for idx, layer in enumerate(self.h):
            adapters = {
                "attention": layer.attention_adapters.adapters,
                "output": layer.output_adapters.adapters,
            }
            for key, adapt in adapters.items():
                if hasattr(adapt, name):
                    if idx not in return_adapters:
                        return_adapters[idx] = {}
                    return_adapters[idx][key] = getattr(adapt, name)

        return return_adapters


class GPT2ModelHeadsMixin(ModelWithFlexibleHeadsAdaptersMixin):
    """Adds flexible heads to a GPT-2 model."""

    head_types = {
        "classification": ClassificationHead,
        "multilabel_classification": MultiLabelClassificationHead,
        "causal_lm": CausalLMHead,
        "tagging": TaggingHead,
    }

    def add_classification_head(
        self,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        overwrite_ok=False,
        multilabel=False,
        id2label=None,
    ):
        """
        Adds a sequence classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
        """

        if multilabel:
            head = MultiLabelClassificationHead(self, head_name, num_labels, layers, activation_function, id2label)
        else:
            head = ClassificationHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)

    def add_causal_lm_head(self, head_name, overwrite_ok=False):
        """
        Adds a causal language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = CausalLMHead(self, head_name)
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)
