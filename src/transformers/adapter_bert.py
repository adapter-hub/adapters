import logging
from typing import Union

import torch

from .adapter_composition import AdapterCompositionBlock, parse_composition
from .adapter_heads import (
    ClassificationHead,
    ModelWithFlexibleHeadsAdaptersMixin,
    MultiLabelClassificationHead,
    MultipleChoiceHead,
    QuestionAnsweringHead,
    TaggingHead,
)
from .adapter_layer_mixin import AdapterLayerBaseMixin
from .adapter_model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


logger = logging.getLogger(__name__)


class BertSelfOutputAdaptersMixin(AdapterLayerBaseMixin):
    """Adds adapters to the BertSelfOutput module."""

    @property
    def adapter_config_key(self):
        return "mh_adapter"


class BertOutputAdaptersMixin(AdapterLayerBaseMixin):
    """Adds adapters to the BertOutput module."""

    @property
    def adapter_config_key(self):
        return "output_adapter"


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module."""

    def add_fusion_layer(self, adapter_names):
        self.attention.output.add_fusion_layer(adapter_names)
        self.output.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.attention.output.add_adapter(adapter_name, layer_idx)
        self.output.add_adapter(adapter_name, layer_idx)

    def enable_adapters(
        self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool
    ):
        self.attention.output.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)
        self.output.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)


class BertEncoderAdaptersMixin:
    """Adds adapters to the BertEncoder module."""

    def add_fusion_layer(self, adapter_names):
        for layer in self.layer:
            layer.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str):
        adapter_config = self.config.adapters.get(adapter_name)
        leave_out = adapter_config.get("leave_out", [])
        for i, layer in enumerate(self.layer):
            if i not in leave_out:
                layer.add_adapter(adapter_name, i)

    def enable_adapters(
        self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool
    ):
        for layer in self.layer:
            layer.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)

    def adjust_attention_mask_for_parallel(self, hidden_states, attention_mask):
        if hidden_states.shape[0] != attention_mask.shape[0]:
            repeats = [1] * len(attention_mask.shape)
            repeats[0] = hidden_states.shape[0] // attention_mask.shape[0]
            attention_mask = attention_mask.repeat(*repeats)
        return attention_mask


class BertModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.encoder.enable_adapters(adapter_setup, True, False)
        self.enable_invertible_adapters(adapter_setup.flatten())
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)

    def train_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.encoder.enable_adapters(adapter_setup, False, True)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)
        # TODO implement fusion for invertible adapters

    def _add_adapter(self, adapter_name):
        self.encoder.add_adapter(adapter_name)
        self.add_invertible_adapter(adapter_name)

    def _add_fusion_layer(self, adapter_names):
        self.encoder.add_fusion_layer(adapter_names)

    def get_fusion_regularization_loss(self):
        reg_loss = 0.0
        target = torch.zeros((self.config.hidden_size, self.config.hidden_size)).fill_diagonal_(1.0).to(self.device)
        for _, v in self.encoder.layer._modules.items():

            for _, layer_fusion in v.output.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

            for _, layer_fusion in v.attention.output.adapter_fusion_layer.items():
                if hasattr(layer_fusion, "value"):
                    reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

        return reg_loss


class BertModelHeadsMixin(ModelWithFlexibleHeadsAdaptersMixin):
    """Adds flexible heads to a BERT-based model class."""

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
        elif config["head_type"] == "tagging":
            self.add_tagging_head(
                head_name,
                config["num_labels"],
                config["layers"],
                config["activation_function"],
                id2label=id2label,
                overwrite_ok=overwrite_ok,
            )
        elif config["head_type"] == "multiple_choice":
            self.add_multiple_choice_head(
                head_name,
                config["num_choices"],
                config["layers"],
                config["activation_function"],
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

    def add_multiple_choice_head(
        self, head_name, num_choices=2, layers=2, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        """
        Adds a multiple choice head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_choices (int, optional): Number of choices. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = MultipleChoiceHead(head_name, num_choices, layers, activation_function, id2label, self)
        self.add_prediction_head(head, overwrite_ok)

    def add_tagging_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        """
        Adds a token classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 1.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = TaggingHead(head_name, num_labels, layers, activation_function, id2label, self)
        self.add_prediction_head(head, overwrite_ok)

    def add_qa_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        head = QuestionAnsweringHead(head_name, num_labels, layers, activation_function, id2label, self)
        self.add_prediction_head(head, overwrite_ok)
