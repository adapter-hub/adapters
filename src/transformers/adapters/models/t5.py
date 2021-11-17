from typing import Union

import torch

from ..composition import AdapterCompositionBlock, parse_composition
from ..heads import Seq2SeqLMHead
from ..layer import AdapterLayerBaseMixin
from ..model_mixin import ModelAdaptersMixin
from .bert import ModelWithFlexibleHeadsAdaptersMixin


class T5SelfAttentionLayerAdaptersMixin(AdapterLayerBaseMixin):
    @property
    def adapter_config_key(self):
        return "mh_adapter"

    @property
    def transformer_layer_norm(self):
        # T5  has layer norms after each component
        return None


class T5CrossAttentionLayerAdaptersMixin(AdapterLayerBaseMixin):
    @property
    def adapter_config_key(self):
        return "cross_adapter"

    @property
    def transformer_layer_norm(self):
        # T5  has layer norms after each component
        return None


class T5FFLayerAdaptersMixin(AdapterLayerBaseMixin):
    @property
    def adapter_config_key(self):
        return "output_adapter"

    @property
    def transformer_layer_norm(self):
        # T5  has layer norms after each component
        return None


class T5BlockAdaptersMixin:
    """Adds adapters to the T5Block module of T5."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def add_fusion_layer(self, adapter_names):
        self.layer[0].add_fusion_layer(adapter_names)  # attention adapters
        self.layer[-1].add_fusion_layer(adapter_names)  # output adapters

    def add_adapter(self, adapter_name: str, layer_idx: int):
        for layer in self.layer:
            layer.add_adapter(adapter_name, layer_idx)

    def enable_adapters(
        self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool
    ):
        for layer in self.layer:
            layer.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)

    def delete_adapter(self, adapter_name):
        for layer in self.layer:
            layer.delete_adapter(adapter_name)

    def delete_fusion_layer(self, adapter_names):
        for layer in self.layer:
            layer.delete_fusion_layer(adapter_names)


class T5StackAdaptersMixin:
    """Adds adapters to the T5Stack module of T5."""

    def point_adapter_configs(self, parent_config):
        self.config = parent_config

    def add_fusion_layer(self, adapter_names):
        for block in self.block:
            block.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, layer_idx_offset: int = 0):
        adapter_config = self.config.adapters.get(adapter_name)
        leave_out = adapter_config.get("leave_out", [])
        for i, block in enumerate(self.block, start=layer_idx_offset):
            if i not in leave_out:
                block.add_adapter(adapter_name, i)

    def delete_adapter(self, adapter_name: str):
        for layer in self.block:
            layer.delete_adapter(adapter_name)

    def delete_fusion_layer(self, adapter_names):
        for layer in self.block:
            layer.delete_fusion_layer(adapter_names)

    def enable_adapters(
        self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool
    ):
        for block in self.block:
            block.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)

    def adjust_attention_mask_for_parallel(self, hidden_states, attention_mask):
        if attention_mask is not None and hidden_states.shape[0] != attention_mask.shape[0]:
            repeats = [1] * len(attention_mask.shape)
            repeats[0] = hidden_states.shape[0] // attention_mask.shape[0]
            attention_mask = attention_mask.repeat(*repeats)
        return attention_mask

    def adjust_tensors_for_parallel(self, hidden_states, *tensors):
        outputs = []
        for tensor in tensors:
            if tensor is not None and hidden_states.shape[0] != tensor.shape[0]:
                repeats = [1] * len(tensor.shape)
                repeats[0] = hidden_states.shape[0] // tensor.shape[0]
                new_tensor = tensor.repeat(*repeats)
                outputs.append(new_tensor)
            else:
                outputs.append(tensor)
        return tuple(outputs)


class T5ModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the T5Model class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        super()._init_adapter_modules()
        if hasattr(self, "encoder"):
            # In T5, the invertible adapters are implemented by the encoder module.
            # Therefore, relay mixin calls to the encoder here.
            self.invertible_adapters = self.encoder.invertible_adapters
            self.add_invertible_adapter = self.encoder.add_invertible_adapter
            self.get_invertible_adapter = self.encoder.get_invertible_adapter
            self.invertible_adapters_forward = self.encoder.invertible_adapters_forward
            self.delete_invertible_adapter = self.encoder.delete_invertible_adapter

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock], train_embeddings=False):
        """Sets the model into mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        if hasattr(self, "encoder"):
            self.encoder.enable_adapters(adapter_setup, True, False)
            self.encoder.enable_invertible_adapters(adapter_setup.flatten())
        self.decoder.enable_adapters(adapter_setup, True, False)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)
        if train_embeddings:
            self.get_input_embeddings().train()

    def train_adapter_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        if hasattr(self, "encoder"):
            self.encoder.enable_adapters(adapter_setup, unfreeze_adapters, True)
        self.decoder.enable_adapters(adapter_setup, unfreeze_adapters, True)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)

    def _add_adapter(self, adapter_name):
        if hasattr(self, "encoder"):
            self.encoder.add_adapter(adapter_name)
            # make sure the layers in encoder & decoder are numbered from 0 to len(encoder+decoder)
            self.decoder.add_adapter(adapter_name, layer_idx_offset=len(self.encoder.block))
        else:
            self.decoder.add_adapter(adapter_name)
        self.encoder.add_invertible_adapter(adapter_name)

    def _add_fusion_layer(self, adapter_names):
        if hasattr(self, "encoder"):
            self.encoder.add_fusion_layer(adapter_names)
        self.decoder.add_fusion_layer(adapter_names)

    def _delete_adapter(self, adapter_name: str):
        if hasattr(self, "encoder"):
            self.encoder.delete_adapter(adapter_name)
            self.encoder.delete_invertible_adapter(adapter_name)
        self.decoder.delete_adapter(adapter_name)

    def _delete_fusion_layer(self, adapter_names):
        if hasattr(self, "encoder"):
            self.encoder.delete_fusion_layer(adapter_names)
        self.decoder.delete_fusion_layer(adapter_names)

    def get_fusion_regularization_loss(self):
        reg_loss = 0.0
        target = torch.zeros((self.config.hidden_size, self.config.hidden_size)).fill_diagonal_(1.0).to(self.device)
        # encoder
        if hasattr(self, "encoder"):
            for _, v in self.encoder.block._modules.items():
                for _, layer_fusion in v.layer[-1].adapter_fusion_layer.items():
                    if hasattr(layer_fusion, "value"):
                        reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

                for _, layer_fusion in v.layer[0].adapter_fusion_layer.items():
                    if hasattr(layer_fusion, "value"):
                        reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()
        # decoder
        for _, v in self.decoder.block._modules.items():
            for _, layer_fusion in v.layer[-1].adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

            for _, layer_fusion in v.layer[0].adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

        return reg_loss

    def get_adapter(self, name):
        return_adapters = {}
        for idx, block in enumerate(self.encoder.block):
            # In each block of T5Stack that is an encoder, the first layer is T5LayerSelfAttention, the second is T5LayerFF
            adapters = {
                "attention": block.layer[0].adapters,
                "output": block.layer[1].adapters,
            }
            for key, adapt in adapters.items():
                if hasattr(adapt, name):
                    if idx not in return_adapters:
                        return_adapters[idx] = {}
                    return_adapters[idx][key] = getattr(adapt, name)

        return return_adapters


class T5ModelHeadsMixin(ModelWithFlexibleHeadsAdaptersMixin):
    """Adds flexible heads to a T5 model."""

    head_types = {
        "seq2seq_lm": Seq2SeqLMHead,
    }

    def add_seq2seq_lm_head(self, head_name, overwrite_ok=False):
        """
        Adds a seq2seq language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = Seq2SeqLMHead(self, head_name)
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)
