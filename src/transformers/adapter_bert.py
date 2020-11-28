# docstyle-ignore-file
import logging
from abc import ABC, abstractmethod
from typing import List, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .adapter_composition import AdapterCompositionBlock, Fuse, Split, Stack, parse_composition
from .adapter_model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin, ModelWithHeadsAdaptersMixin
from .adapter_modeling import Activation_Function_Class, Adapter, BertFusion
from .modeling_outputs import (
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


logger = logging.getLogger(__name__)


def get_fusion_regularization_loss(model):
    if hasattr(model, "base_model"):
        model = model.base_model
    elif hasattr(model, "encoder"):
        pass
    else:
        raise Exception("Model not passed correctly, please pass a transformer model with an encoder")

    reg_loss = 0.0
    target = torch.zeros((model.config.hidden_size, model.config.hidden_size)).fill_diagonal_(1.0).to(model.device)
    for k, v in model.encoder.layer._modules.items():

        for _, layer_fusion in v.output.adapter_fusion_layer.items():
            if hasattr(layer_fusion, "value"):
                reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

        for _, layer_fusion in v.attention.output.adapter_fusion_layer.items():
            if hasattr(layer_fusion, "value"):
                reg_loss += 0.01 * (target - layer_fusion.value.weight).pow(2).sum()

    return reg_loss


class BertAdaptersBaseMixin(ABC):
    """An abstract base implementation of adapter integration into a Transformer block.
    In BERT, subclasses of this module are placed in the BertSelfOutput module and in the BertOutput module.
    """

    # override this property if layer norm has a different name
    @property
    def layer_norm(self):
        return self.LayerNorm

    @property
    @abstractmethod
    def adapter_modules(self):
        """Gets the module dict holding the adapter modules available in this block."""
        pass

    @property
    @abstractmethod
    def adapter_config_key(self):
        """Gets the name of the key by which this adapter location is identified in the adapter configuration."""
        pass

    def _init_adapter_modules(self):
        self.adapter_fusion_layer = nn.ModuleDict(dict())

    def add_adapter(self, adapter_name: str):
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config[self.adapter_config_key]:
            adapter = Adapter(
                input_size=self.config.hidden_size,
                down_sample=self.config.hidden_size // adapter_config["reduction_factor"],
                add_layer_norm_before=adapter_config["ln_before"],
                add_layer_norm_after=adapter_config["ln_after"],
                non_linearity=adapter_config["non_linearity"],
                residual_before_ln=adapter_config["adapter_residual_before_ln"],
            )
            self.adapter_modules[adapter_name] = adapter

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        """See BertModel.add_fusion_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        if self.config.adapters.common_config_value(adapter_names, self.adapter_config_key):
            self.adapter_fusion_layer[",".join(adapter_names)] = BertFusion(self.config)

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        """Unfreezes a given list of adapters, the adapter fusion layer, or both

        :param adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
        :param unfreeze_adapters: whether the adapters themselves should be unfreezed
        :param unfreeze_fusion: whether the adapter attention layer for the given adapters should be unfreezed
        """
        if unfreeze_adapters:
            for adapter_name in adapter_setup.flatten():
                if adapter_name in self.adapter_modules:
                    for param in self.adapter_modules[adapter_name].parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_setup, Fuse):
                if adapter_setup.name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[adapter_setup.name].parameters():
                        param.requires_grad = True
            for sub_setup in adapter_setup:
                if isinstance(sub_setup, Fuse):
                    if sub_setup.name in self.adapter_fusion_layer:
                        for param in self.adapter_fusion_layer[sub_setup.name].parameters():
                            param.requires_grad = True

    def get_adapter_preparams(
        self,
        adapter_config,
        hidden_states,
        input_tensor,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration
        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None

        if adapter_config["residual_before_ln"]:
            residual = hidden_states

        if hasattr(self.config, "adapter_fusion") and self.config.adapter_fusion["query_before_ln"]:
            query = hidden_states

        if adapter_config["original_ln_before"]:
            hidden_states = self.layer_norm(hidden_states + input_tensor)

        if not adapter_config["residual_before_ln"]:
            residual = hidden_states

        if hasattr(self.config, "adapter_fusion") and not self.config.adapter_fusion["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def adapter_stack(self, adapter_setup: Stack, hidden_states, input_tensor):
        for adapter_stack_layer in adapter_setup:
            # Case 1: We have a nested fusion layer -> call fusion method
            if isinstance(adapter_stack_layer, Fuse):
                hidden_states = self.adapter_fusion(adapter_stack_layer, hidden_states, input_tensor)
                up = hidden_states  # TODO
            # Case 2: We have a nested split layer -> call split method
            elif isinstance(adapter_stack_layer, Split):
                hidden_states = self.adapter_split(adapter_stack_layer, hidden_states, input_tensor)
                up = hidden_states  # TODO
            # Case 3: We have a single adapter which is part of this module -> forward pass
            elif adapter_stack_layer in self.adapter_modules:
                adapter_layer = self.adapter_modules[adapter_stack_layer]
                adapter_config = self.config.adapters.get(adapter_stack_layer)
                hidden_states, _, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)
                hidden_states, _, up = adapter_layer(hidden_states, residual_input=residual)
            # Case X: No adapter which is part of this module -> ignore

        return hidden_states, up

    def adapter_fusion(self, adapter_setup: Fuse, hidden_states, input_tensor):
        # config of _last_ fused adapter is significant
        adapter_config = self.config.adapters.get(adapter_setup.last())
        hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        up_list = []

        for adapter_block in adapter_setup:
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                _, up = self.adapter_stack(adapter_block, hidden_states, input_tensor)
            # Case 2: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapter_modules:
                adapter_layer = self.adapter_modules[adapter_block]
                _, _, up = adapter_layer(hidden_states, residual_input=residual)
            # Case X: No adapter which is part of this module -> ignore
            up_list.append(up)

        if len(up_list) > 0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)

            fusion_name = ",".join(adapter_setup)

            hidden_states = self.adapter_fusion_layer[fusion_name](
                query,
                up_list,
                up_list,
                residual,
            )

        return hidden_states

    def adapter_split(self, adapter_setup: Split, hidden_states, input_tensor):
        # config of _first_ of splitted adapters is significant
        adapter_config = self.config.adapters.get(adapter_setup.first())
        hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        first_hidden_states = hidden_states[:, : adapter_setup.split_index, :]
        second_hidden_states = hidden_states[:, adapter_setup.split_index :, :]
        first_residual = residual[:, : adapter_setup.split_index, :]
        second_residual = residual[:, adapter_setup.split_index :, :]

        if adapter_setup.left in self.adapter_modules:
            first_hidden_states = self.adapter_modules[adapter_setup.left](
                first_hidden_states, residual_input=first_residual
            )
        if adapter_setup.right in self.adapter_modules:
            second_hidden_states = self.adapter_modules[adapter_setup.right](
                second_hidden_states, residual_input=second_residual
            )

        hidden_states = torch.cat((first_hidden_states, second_hidden_states), dim=1)
        return hidden_states

    def adapters_forward(self, hidden_states, input_tensor):
        adapter_setup = self.config.adapters.active_setup if hasattr(self.config, "adapters") else None
        if adapter_setup is not None and (len(set(self.adapter_modules.keys()) & adapter_setup.flatten()) > 0):
            if isinstance(adapter_setup, Stack):
                hidden_states, _ = self.adapter_stack(adapter_setup, hidden_states, input_tensor)
            elif isinstance(adapter_setup, Fuse):
                hidden_states = self.adapter_fusion(adapter_setup, hidden_states, input_tensor)
            elif isinstance(adapter_setup, Split):
                hidden_states = self.adapter_split(adapter_setup, hidden_states, input_tensor)
            else:
                raise ValueError(f"Invalid adapter setup {adapter_setup}")

            last_config = self.config.adapters.get(adapter_setup.last())
            if last_config["original_ln_after"]:
                hidden_states = self.layer_norm(hidden_states + input_tensor)

        else:
            hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class BertSelfOutputAdaptersMixin(BertAdaptersBaseMixin):
    """Adds adapters to the BertSelfOutput module."""

    @property
    def adapter_modules(self):
        return self.attention_adapters

    @property
    def adapter_config_key(self):
        return "mh_adapter"

    def _init_adapter_modules(self):
        super()._init_adapter_modules()
        self.attention_adapters = nn.ModuleDict(dict())


class BertOutputAdaptersMixin(BertAdaptersBaseMixin):
    """Adds adapters to the BertOutput module."""

    @property
    def adapter_modules(self):
        return self.output_adapters

    @property
    def adapter_config_key(self):
        return "output_adapter"

    def _init_adapter_modules(self):
        super()._init_adapter_modules()
        self.output_adapters = nn.ModuleDict(dict())


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module."""

    def add_fusion_layer(self, adapter_names):
        self.attention.output.add_fusion_layer(adapter_names)
        self.output.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str):
        self.attention.output.add_adapter(adapter_name)
        self.output.add_adapter(adapter_name)

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool):
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
                layer.add_adapter(adapter_name)

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_attention: bool):
        for layer in self.layer:
            layer.enable_adapters(adapter_setup, unfreeze_adapters, unfreeze_attention)


class BertModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the BertModel module."""

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

    def add_adapter(self, adapter_name: str, config=None):
        """Adds a new adapter module of the specified type to the model.

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
        self.encoder.add_fusion_layer(adapter_names)


class BertModelHeadsMixin(ModelWithHeadsAdaptersMixin):
    """Adds heads to a Bert-based module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.active_head = None

    def _init_head_modules(self):
        if not hasattr(self.config, "prediction_heads"):
            self.config.prediction_heads = {}
        self.heads = nn.ModuleDict(dict())
        # add modules for heads in config
        for head_name in self.config.prediction_heads:
            self._add_prediction_head_module(head_name)

    @property
    def active_head(self):
        return self._active_head

    @active_head.setter
    def active_head(self, head_name):
        self._active_head = head_name
        if head_name is not None and head_name in self.config.prediction_heads:
            self.config.label2id = self.config.prediction_heads[head_name]["label2id"]
            self.config.id2label = self.get_labels_dict(head_name)

    def set_active_adapters(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """Sets the adapter modules to be used by default in every forward pass.
        This setting can be overriden by passing the `adapter_names` parameter in the `foward()` pass.
        If no adapter with the given name is found, no module of the respective type will be activated.
        In case the calling model class supports named prediction heads, this method will attempt to activate a prediction head with the name of the last adapter in the list of passed adapter names.

        Args:
            adapter_setup (list): The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        self.base_model.set_active_adapters(adapter_setup)
        # use last adapter name as name of prediction head
        if self.active_adapters:
            head_name = self.active_adapters.last()
            if head_name in self.config.prediction_heads:
                self.active_head = head_name

            else:
                logger.info("No prediction head for task_name '{}' available.".format(head_name))

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
        """Adds a sequence classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
        """
        if multilabel:
            head_type = "multilabel_classification"
        else:
            head_type = "classification"
        config = {
            "head_type": head_type,
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_multiple_choice_head(
        self, head_name, num_choices=2, layers=2, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        """Adds a multiple choice head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_choices (int, optional): Number of choices. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        config = {
            "head_type": "multiple_choice",
            "num_choices": num_choices,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_tagging_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        """Adds a token classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 1.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        config = {
            "head_type": "tagging",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_qa_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False, id2label=None
    ):
        config = {
            "head_type": "question_answering",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_prediction_head(
        self,
        head_name,
        config,
        overwrite_ok=False,
    ):
        if head_name not in self.config.prediction_heads or overwrite_ok:
            self.config.prediction_heads[head_name] = config

            if "label2id" not in config.keys() or config["label2id"] is None:
                if "num_labels" in config.keys():
                    config["label2id"] = {"LABEL_" + str(num): num for num in range(config["num_labels"])}
                if "num_choices" in config.keys():
                    config["label2id"] = {"LABEL_" + str(num): num for num in range(config["num_choices"])}

            logger.info(f"Adding head '{head_name}' with config {config}.")
            self._add_prediction_head_module(head_name)
            self.active_head = head_name

        else:
            raise ValueError(
                f"Model already contains a head with name '{head_name}'. Use overwrite_ok=True to force overwrite."
            )

    def _add_prediction_head_module(self, head_name):
        head_config = self.config.prediction_heads.get(head_name)

        pred_head = []
        for l in range(head_config["layers"]):
            pred_head.append(nn.Dropout(self.config.hidden_dropout_prob))
            if l < head_config["layers"] - 1:
                pred_head.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
                pred_head.append(Activation_Function_Class(head_config["activation_function"]))
            else:
                if "num_labels" in head_config:
                    pred_head.append(nn.Linear(self.config.hidden_size, head_config["num_labels"]))
                else:  # used for multiple_choice head
                    pred_head.append(nn.Linear(self.config.hidden_size, 1))

        self.heads[head_name] = nn.Sequential(*pred_head)

        self.heads[head_name].apply(self._init_weights)
        self.heads[head_name].train(self.training)  # make sure training mode is consistent

    def forward_head(self, outputs, head_name=None, attention_mask=None, labels=None, return_dict=False):
        head_name = head_name or self.active_head
        if not head_name:
            logger.debug("No prediction head is used.")
            return outputs

        if head_name not in self.config.prediction_heads:
            raise ValueError("Unknown head_name '{}'".format(head_name))

        head = self.config.prediction_heads[head_name]

        sequence_output = outputs[0]
        loss = None

        if head["head_type"] == "classification":
            logits = self.heads[head_name](sequence_output[:, 0])

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                if head["num_labels"] == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, head["num_labels"]), labels.view(-1))
                outputs = (loss,) + outputs

            if return_dict:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return outputs

        elif head["head_type"] == "multilabel_classification":
            logits = self.heads[head_name](sequence_output[:, 0])

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                loss_fct = BCEWithLogitsLoss()
                if labels.dtype != torch.float32:
                    labels = labels.float()
                loss = loss_fct(logits, labels)
                outputs = (loss,) + outputs

            if return_dict:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return outputs

        elif head["head_type"] == "multiple_choice":
            logits = self.heads[head_name](sequence_output[:, 0])
            logits = logits.view(-1, head["num_choices"])

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                outputs = (loss,) + outputs

            if return_dict:
                return MultipleChoiceModelOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return outputs

        elif head["head_type"] == "tagging":
            logits = self.heads[head_name](sequence_output)

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

            if return_dict:
                return TokenClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return outputs

        elif head["head_type"] == "question_answering":
            logits = self.heads[head_name](sequence_output)

            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (
                start_logits,
                end_logits,
            ) + outputs[2:]
            if labels is not None:
                start_positions, end_positions = labels
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

            if return_dict:
                return QuestionAnsweringModelOutput(
                    loss=loss,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return outputs

        else:
            raise ValueError("Unknown head_type '{}'".format(head["head_type"]))

    def get_labels_dict(self, head_name=None):
        """
        Returns the id2label dict for the given head
        Args:
            head_name: (str, optional) the name of the head which labels should be returned. Default is None.
            If the name is None the labels of the active head are returned

        Returns: id2label

        """
        if head_name is None:
            head_name = self.active_head
        if head_name is None:
            raise ValueError("No head name given and no active head in the model")
        if "label2id" in self.config.prediction_heads[head_name].keys():
            return {id_: label for label, id_ in self.config.prediction_heads[head_name]["label2id"].items()}
        else:
            return None

    def get_labels(self, head_name=None):
        """
        Returns the labels the given head is assigning/predicting
        Args:
            head_name: (str, optional) the name of the head which labels should be returned. Default is None.
            If the name is None the labels of the active head are returned

        Returns: labels

        """
        label_dict = self.get_labels_dict(head_name)
        if label_dict is None:
            return None
        else:
            return list(label_dict.values())
