from typing import Union

import torch
from torch import nn

from .adapter_bert import BertAdaptersBaseMixin
from .adapter_composition import AdapterCompositionBlock, parse_composition
from .adapter_heads import (
    ClassificationHead,
    ModelWithFlexibleHeadsAdaptersMixin,
    MultiLabelClassificationHead,
    QuestionAnsweringHead,
)
from .adapter_model_mixin import ModelAdaptersMixin


class BartSelfAttentionAdaptersModule(BertAdaptersBaseMixin, nn.Module):
    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def adapter_config_key(self):
        return "mh_adapter"

    @property
    def layer_norm(self):
        # if layer norm was applied before attention/ FFW block, don't use it here
        if self.config.normalize_before:
            return None
        else:
            return self.parent.self_attn_layer_norm


class BartCrossAttentionAdaptersModule(BertAdaptersBaseMixin, nn.Module):
    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def adapter_config_key(self):
        return "cross_adapter"

    @property
    def layer_norm(self):
        # if layer norm was applied before attention/ FFW block, don't use it here
        if self.config.normalize_before:
            return None
        else:
            return self.parent.encoder_attn_layer_norm


class BartOutputAdaptersModule(BertAdaptersBaseMixin, nn.Module):
    def __init__(self, parent):
        super().__init__()
        # keep a reference to the parent module without registering as a submodule
        object.__setattr__(self, "parent", parent)
        self.config = parent.config

    @property
    def adapter_config_key(self):
        return "output_adapter"

    @property
    def layer_norm(self):
        # if layer norm was applied before attention/ FFW block, don't use it here
        if self.config.normalize_before:
            return None
        else:
            return self.parent.final_layer_norm


class BartEncoderLayerAdaptersMixin:
    """Adds adapters to the BartEncoderLayer or BartDecoderLayer module of BART."""

    def _init_adapter_modules(self):
        self.attention_adapters = BartSelfAttentionAdaptersModule(self)
        self.output_adapters = BartOutputAdaptersModule(self)
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


class BartDecoderLayerAdaptersMixin(BartEncoderLayerAdaptersMixin):
    """Adds adapters to the BartEncoderLayer or BartDecoderLayer module of BART."""

    def _init_adapter_modules(self):
        super()._init_adapter_modules()
        self.cross_attention_adapters = BartCrossAttentionAdaptersModule(self)
        self.cross_attention_adapters._init_adapter_modules()

    def add_fusion_layer(self, adapter_names):
        super().add_fusion_layer(adapter_names)
        self.cross_attention_adapters.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str):
        super().add_adapter(adapter_name)
        self.cross_attention_adapters.add_adapter(adapter_name)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        super().enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)
        self.cross_attention_adapters.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)


class BartEncoderDecoderAdaptersMixin:
    """Adds adapters to the BartEncoder or BartDecoder module."""

    def add_fusion_layer(self, adapter_names):
        for layer in self.layers:
            layer.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str):
        adapter_config = self.config.adapters.get(adapter_name)
        leave_out = adapter_config.get("leave_out", [])
        for i, layer in enumerate(self.layers):
            if i not in leave_out:
                layer.add_adapter(adapter_name)

    def enable_adapters(
        self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool
    ):
        for layer in self.layers:
            layer.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)


class BartModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the DistilBert module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.encoder.enable_adapters(adapter_setup, True, False)
        self.decoder.enable_adapters(adapter_setup, True, False)
        # self.enable_invertible_adapters(adapter_setup.flatten())
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)

    def train_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.encoder.enable_adapters(adapter_setup, False, True)
        self.decoder.enable_adapters(adapter_setup, False, True)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)

    def _add_adapter(self, adapter_name):
        self.encoder.add_adapter(adapter_name)
        self.decoder.add_adapter(adapter_name)
        # self.add_invertible_adapter(adapter_name)

    def _add_fusion_layer(self, adapter_names):
        self.encoder.add_fusion_layer(adapter_names)
        self.decoder.add_fusion_layer(adapter_names)

    def get_fusion_regularization_loss(self):
        reg_loss = 0.0
        target = torch.zeros((self.config.hidden_size, self.config.hidden_size)).fill_diagonal_(1.0).to(self.device)
        # encoder
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


class BartModelHeadsMixin(ModelWithFlexibleHeadsAdaptersMixin):
    """Adds flexible heads to a BART model."""

    def add_prediction_head_from_config(self, head_name, config, overwrite_ok=False):
        id2label = (
            {id_: label for label, id_ in config["label2id"].items()}
            if "label2id" in config.keys() and config["label2id"]
            else None
        )
        if config["head_type"] == "classification":
            self.add_classification_head(
                head_name,
                config["num_labels"],
                config["layers"],
                config["activation_function"],
                id2label=id2label,
                overwrite_ok=overwrite_ok,
            )
        elif config["head_type"] == "multilabel_classification":
            self.add_classification_head(
                head_name,
                config["num_labels"],
                config["layers"],
                config["activation_function"],
                multilabel=True,
                id2label=id2label,
                overwrite_ok=overwrite_ok,
            )
        elif config["head_type"] == "question_answering":
            self.add_qa_head(
                head_name,
                config["num_labels"],
                config["layers"],
                config["activation_function"],
                id2label=id2label,
                overwrite_ok=overwrite_ok,
            )
        else:
            if config["head_type"] in self.config.custom_heads:
                self.add_custom_head(head_name, config, overwrite_ok=overwrite_ok)
            else:
                raise AttributeError("Please register the PredictionHead before loading the model")

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
            head = MultiLabelClassificationHead(head_name, num_labels, layers, activation_function, id2label, self)
        else:
            head = ClassificationHead(head_name, num_labels, layers, activation_function, id2label, self)
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
        head = QuestionAnsweringHead(head_name, num_labels, layers, activation_function, id2label, self)
        self.add_prediction_head(head, overwrite_ok)
