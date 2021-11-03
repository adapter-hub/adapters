from typing import Union

import torch
from torch import nn

from ..composition import AdapterCompositionBlock, parse_composition
from ..heads import (
    ClassificationHead,
    ModelWithFlexibleHeadsAdaptersMixin,
    MultiLabelClassificationHead,
    QuestionAnsweringHead,
    Seq2SeqLMHead,
)
from ..layer import AdapterLayerBaseMixin
from ..model_mixin import ModelAdaptersMixin


class BartSelfAttentionAdaptersModule(AdapterLayerBaseMixin, nn.Module):
    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def adapter_config_key(self):
        return "mh_adapter"

    @property
    def transformer_layer_norm(self):
        # MBart has layer norms before each component
        if self.config.model_type == "mbart":
            return None
        else:
            return self.parent.self_attn_layer_norm


class BartCrossAttentionAdaptersModule(AdapterLayerBaseMixin, nn.Module):
    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def adapter_config_key(self):
        return "cross_adapter"

    @property
    def transformer_layer_norm(self):
        # MBart has layer norms before each component
        if self.config.model_type == "mbart":
            return None
        else:
            return self.parent.encoder_attn_layer_norm


class BartOutputAdaptersModule(AdapterLayerBaseMixin, nn.Module):
    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def adapter_config_key(self):
        return "output_adapter"

    @property
    def transformer_layer_norm(self):
        # MBart has layer norms before each component
        if self.config.model_type == "mbart":
            return None
        else:
            return self.parent.final_layer_norm


class BartEncoderLayerAdaptersMixin:
    """Adds adapters to the BartEncoderLayer module of BART."""

    def _init_adapter_modules(self):
        self.attention_adapters = BartSelfAttentionAdaptersModule(self)
        self.output_adapters = BartOutputAdaptersModule(self)
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


class BartDecoderLayerAdaptersMixin(BartEncoderLayerAdaptersMixin):
    """Adds adapters to the BartDecoderLayer module of BART."""

    def _init_adapter_modules(self):
        super()._init_adapter_modules()
        self.cross_attention_adapters = BartCrossAttentionAdaptersModule(self)
        self.cross_attention_adapters._init_adapter_modules()

    def add_fusion_layer(self, adapter_names):
        super().add_fusion_layer(adapter_names)
        self.cross_attention_adapters.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, layer_idx: int):
        super().add_adapter(adapter_name, layer_idx)
        self.cross_attention_adapters.add_adapter(adapter_name, layer_idx)

    def delete_adapter(self, adapter_name):
        super().delete_adapter(adapter_name)
        self.cross_attention_adapters.delete_adapter(adapter_name)

    def delete_fusion_layer(self, adapter_names):
        super().delete_fusion_layer(adapter_names)
        self.cross_attention_adapters.delete_fusion_layer(adapter_names)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        super().enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)
        self.cross_attention_adapters.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)

    # Makes sure the "parent" reference always points to the correct module.
    # This is especially relevant when using torch data parallelism.
    @staticmethod
    def _adapter_block_pre_hook(module, input_tensors):
        object.__setattr__(module.attention_adapters, "parent", module)
        object.__setattr__(module.output_adapters, "parent", module)
        object.__setattr__(module.cross_attention_adapters, "parent", module)


class BartEncoderDecoderAdaptersMixin:
    """Adds adapters to the BartEncoder or BartDecoder module."""

    def add_fusion_layer(self, adapter_names):
        for layer in self.layers:
            layer.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, layer_idx_offset: int = 0):
        adapter_config = self.config.adapters.get(adapter_name)
        leave_out = adapter_config.get("leave_out", [])
        for i, layer in enumerate(self.layers, start=layer_idx_offset):
            if i not in leave_out:
                layer.add_adapter(adapter_name, i)

    def delete_adapter(self, adapter_name: str):
        for layer in self.layers:
            layer.delete_adapter(adapter_name)

    def delete_fusion_layer(self, adapter_names):
        for layer in self.layers:
            layer.delete_fusion_layer(adapter_names)

    def enable_adapters(
        self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool
    ):
        for layer in self.layers:
            layer.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)

    def adjust_attention_mask_for_parallel(self, hidden_states, attention_mask):
        if attention_mask is not None and hidden_states.shape[0] != attention_mask.shape[0]:
            repeats = [1] * len(attention_mask.shape)
            repeats[0] = hidden_states.shape[0] // attention_mask.shape[0]
            attention_mask = attention_mask.repeat(*repeats)
        return attention_mask


class BartModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the BartModel class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        super()._init_adapter_modules()
        if hasattr(self, "encoder"):
            # In BART, the invertible adapters are implemented by the encoder module.
            # Therefore, relay mixin calls to the encoder here.
            self.invertible_adapters = self.encoder.invertible_adapters
            self.add_invertible_adapter = self.encoder.add_invertible_adapter
            self.get_invertible_adapter = self.encoder.get_invertible_adapter
            self.invertible_adapters_forward = self.encoder.invertible_adapters_forward

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
            self.decoder.add_adapter(adapter_name, layer_idx_offset=len(self.encoder.layers))
            self.encoder.add_invertible_adapter(adapter_name)
        else:
            self.decoder.add_adapter(adapter_name)

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
            for _, v in self.encoder.layers._modules.items():
                for _, layer_fusion in v.output_adapters.adapter_fusion_layer.items():
                    if hasattr(layer_fusion, "value"):
                        reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

                for _, layer_fusion in v.attention_adapters.adapter_fusion_layer.items():
                    if hasattr(layer_fusion, "value"):
                        reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()
        # decoder
        for _, v in self.decoder.layers._modules.items():
            for _, layer_fusion in v.output_adapters.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

            for _, layer_fusion in v.attention_adapters.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

        return reg_loss

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

    def get_adapter(self, name):
        return_adapters = {}
        for idx, layer in enumerate(self.encoder.layers):
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


class BartModelHeadsMixin(ModelWithFlexibleHeadsAdaptersMixin):
    """
    Adds flexible heads to a BART model.
    """

    head_types = {
        "classification": ClassificationHead,
        "multilabel_classification": MultiLabelClassificationHead,
        "question_answering": QuestionAnsweringHead,
        "seq2seq_lm": Seq2SeqLMHead,
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

    def add_qa_head(
        self,
        head_name,
        num_labels=2,
        layers=1,
        activation_function="tanh",
        overwrite_ok=False,
        id2label=None,
    ):
        head = QuestionAnsweringHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)

    def add_seq2seq_lm_head(
        self,
        head_name,
        overwrite_ok=False,
    ):
        """
        Adds a sequence-to-sequence language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = Seq2SeqLMHead(self, head_name)
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)
