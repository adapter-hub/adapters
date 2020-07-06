import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .adapter_config import DEFAULT_ADAPTER_CONFIG, AdapterType
from .adapter_model_mixin import ModelAdaptersMixin, ModelWithHeadsAdaptersMixin
from .adapter_modeling import (
    Activation_Function_Class,
    Adapter,
    AdapterFusionSentLvlDynamic,
    AdapterWeightingSentLvl,
    AdapterWeightingSentLvlDynamic,
    BertAdapterAttention,
    GLOWCouplingBlock,
    NICECouplingBlock,
    SimpleAdapterWeightingStatic,
)


logger = logging.getLogger(__name__)


class BertSelfOutputAdaptersMixin:
    """Adds adapters to the BertSelfOutput module.
    """

    def _init_adapter_modules(self):
        self.attention_text_task_adapters = nn.ModuleDict(dict())
        self.attention_adapters_fusion = nn.ModuleDict(dict())
        self.attention_text_lang_adapters = nn.ModuleDict(dict())
        self.language_attention_adapters_fusion = nn.ModuleDict(dict())
        self.language_adapter_attention = nn.ModuleDict(dict())

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config["mh_adapter"]:
            adapter = Adapter(
                input_size=self.config.hidden_size,
                down_sample=self.config.hidden_size // adapter_config["reduction_factor"],
                add_layer_norm_before=adapter_config["ln_before"],
                add_layer_norm_after=adapter_config["ln_after"],
                non_linearity=adapter_config["non_linearity"],
                residual_before_ln=adapter_config["adapter_residual_before_ln"],
            )
            if adapter_type == AdapterType.text_task:
                self.attention_text_task_adapters[adapter_name] = adapter
            elif adapter_type == AdapterType.text_lang:
                self.attention_text_lang_adapters[adapter_name] = adapter
            else:
                raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def add_attention_layer(self, tasks):
        """See BertModel.add_attention_layer"""
        task_names = tasks if isinstance(tasks, list) else tasks.split("_")
        adapter_config = self.config.adapters.common_config(task_names)
        if not adapter_config:
            raise ValueError("All tasks used in the attention layer must have the same configuration.")
        if adapter_config["mh_adapter"]:
            if adapter_config["attention_type"] == "tok-lvl":
                layer = BertAdapterAttention(self.config)
            elif adapter_config["attention_type"] == "sent-lvl":
                layer = AdapterWeightingSentLvl(self.config, len(task_names))
            elif adapter_config["attention_type"] == "sent-lvl-dynamic":
                layer = AdapterWeightingSentLvlDynamic(self.config, len(task_names))
            elif adapter_config["attention_type"] == "static":
                layer = SimpleAdapterWeightingStatic(self.config, len(task_names))
            else:
                raise Exception("Unknown attention type: {}".format(adapter_config["attention_type"]))

            self.attention_adapters_fusion["_".join(task_names)] = layer

            if adapter_config["new_attention_norm"]:
                self.attention_layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

    def enable_adapters(self, adapter_type: AdapterType, unfreeze_adapters: bool, unfreeze_attention: bool):
        # TODO cleanup?
        if adapter_type == AdapterType.text_task:
            if unfreeze_adapters:
                for param in self.attention_text_task_adapters.parameters():
                    param.requires_grad = True
            if unfreeze_attention:
                for param in self.attention_adapters_fusion.parameters():
                    param.requires_grad = True

                for adap in self.attention_text_task_adapters.values():
                    for param in adap.adapter_attention.parameters():
                        param.requires_grad = True

                if hasattr(self, "attention_layer_norm"):
                    for param in self.attention_layer_norm.parameters():
                        param.requires_grad = True
        elif adapter_type == AdapterType.text_lang:
            if unfreeze_adapters:
                for param in self.attention_text_lang_adapters.parameters():
                    param.requires_grad = True
            if unfreeze_attention:
                for param in self.language_attention_adapters_fusion.parameters():
                    param.requires_grad = True

                for adap in self.attention_text_lang_adapters.values():
                    for param in adap.language_adapter_attention.parameters():
                        param.requires_grad = True

                if hasattr(self, "language_attention_layer_norm"):
                    for param in self.language_attention_layer_norm.parameters():
                        param.requires_grad = True
        else:
            raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def adapters_forward(self, hidden_states, input_tensor, tasks=None, language=None):
        adapter_used = False

        # Language adapter
        if language:
            lang_adapter_config = self.config.adapters.get(language)
            if lang_adapter_config and language in self.attention_text_lang_adapters:
                adapter_used = True

                if lang_adapter_config["residual_before_ln"]:
                    residual = hidden_states

                if lang_adapter_config["original_ln_before"]:
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)

                if not lang_adapter_config["residual_before_ln"]:
                    residual = hidden_states

                hidden_states, adapter_attention, down, up = self.attention_text_lang_adapters[language](
                    hidden_states, residual_input=residual
                )
                if lang_adapter_config["original_ln_after"]:
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # Task adapters
        # filter tasks that are available in this module
        if tasks:
            tasks = [t for t in tasks if t in self.attention_text_task_adapters]
        if tasks:
            # if we have multiple tasks and use fusion, all configs are assumed to be equal
            task_adapter_config = self.config.adapters.get(tasks[0])
            adapter_used = True

            if task_adapter_config["residual_before_ln"]:
                residual = hidden_states

            # if hasattr(self.config, "fusion_config") and self.config.fusion_config["query_before_ln"]:
            #     query = hidden_states

            if task_adapter_config["original_ln_before"]:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

            if not task_adapter_config["residual_before_ln"]:
                residual = hidden_states

            # if hasattr(self.config, "fusion_config") and not self.config.fusion_config["query_before_ln"]:
            #     query = hidden_states

            # if we have multiple tasks, use fusion
            if len(tasks) > 1:
                # TODO see BertOutput module
                raise NotImplementedError()

            # otherwise, only use one task adapter without attention
            else:
                hidden_states, adapter_attention, down, up = self.attention_text_task_adapters[tasks[0]](
                    hidden_states, residual_input=residual
                )
                if task_adapter_config["original_ln_after"]:
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # In case we haven't used any adapter
        if not adapter_used:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertOutputAdaptersMixin:
    """Adds adapters to the BertOutput module.
    """

    def _init_adapter_modules(self):
        # self.bert_adapter_att = BertAdapterAttention(config)
        # self.bert_adapter_att = SimpleAdapterWeightingSentLvl(config)
        self.bert_adapter_att = nn.ModuleDict(dict())
        self.layer_text_task_adapters = nn.ModuleDict(dict())
        self.bert_language_adapter_att = nn.ModuleDict(dict())
        self.layer_text_lang_adapters = nn.ModuleDict(dict())

    def add_attention_layer(self, tasks):
        """See BertModel.add_attention_layer"""
        task_names = tasks if isinstance(tasks, list) else tasks.split("_")
        adapter_config = self.config.adapters.common_config(task_names)
        if not adapter_config:
            raise ValueError("All tasks used in the attention layer must have the same configuration.")
        if adapter_config["output_adapter"]:
            if adapter_config["attention_type"] == "tok-lvl":
                layer = BertAdapterAttention(self.config)
            elif adapter_config["attention_type"] == "sent-lvl":
                layer = AdapterWeightingSentLvl(self.config, len(task_names))
            elif adapter_config["attention_type"] == "sent-lvl-dynamic":
                layer = AdapterWeightingSentLvlDynamic(self.config, len(task_names))
            elif adapter_config["attention_type"] == "static":
                layer = SimpleAdapterWeightingStatic(self.config, len(task_names))
            elif adapter_config["attention_type"] == "sent-lvl-fusion":
                layer = AdapterFusionSentLvlDynamic(self.config, len(task_names))

            else:
                raise Exception("Unknown attention type: {}".format(adapter_config["attention_type"]))

            self.bert_adapter_att["_".join(task_names)] = layer

            if adapter_config["new_attention_norm"]:
                self.attention_layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config["output_adapter"]:
            adapter = Adapter(
                input_size=self.config.hidden_size,
                down_sample=self.config.hidden_size // adapter_config["reduction_factor"],
                add_layer_norm_before=adapter_config["ln_before"],
                add_layer_norm_after=adapter_config["ln_after"],
                non_linearity=adapter_config["non_linearity"],
                residual_before_ln=adapter_config["adapter_residual_before_ln"],
            )
            if adapter_type == AdapterType.text_task:
                self.layer_text_task_adapters[adapter_name] = adapter
            elif adapter_type == AdapterType.text_lang:
                self.layer_text_lang_adapters[adapter_name] = adapter
            else:
                raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def enable_adapters(self, adapter_type: AdapterType, unfreeze_adapters: bool, unfreeze_attention: bool):
        # TODO cleanup?
        if adapter_type == AdapterType.text_task:
            if unfreeze_adapters:
                for param in self.layer_text_task_adapters.parameters():
                    param.requires_grad = True
            if unfreeze_attention:
                for adap in self.layer_text_task_adapters.values():
                    for param in adap.adapter_attention.parameters():
                        param.requires_grad = True

                for param in self.bert_adapter_att.parameters():
                    param.requires_grad = True

                if hasattr(self, "attention_layer_norm"):
                    for param in self.attention_layer_norm.parameters():
                        param.requires_grad = True
        elif adapter_type == AdapterType.text_lang:
            if unfreeze_adapters:
                for param in self.layer_text_lang_adapters.parameters():
                    param.requires_grad = True
            if unfreeze_attention:
                for adap in self.layer_text_lang_adapters.values():
                    for param in adap.adapter_attention.parameters():
                        param.requires_grad = True

                for param in self.bert_language_adapter_att.parameters():
                    param.requires_grad = True

                if hasattr(self, "language_attention_layer_norm"):
                    for param in self.language_attention_layer_norm.parameters():
                        param.requires_grad = True
        else:
            raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def adapters_forward(self, hidden_states, input_tensor, attention_mask, tasks=None, language=None):
        adapter_used = False

        # Language adapter
        if language:
            lang_adapter_config = self.config.adapters.get(language)
            if lang_adapter_config and language in self.layer_text_lang_adapters:
                adapter_used = True

                if lang_adapter_config["residual_before_ln"]:
                    residual = hidden_states

                if lang_adapter_config["original_ln_before"]:
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)

                if not lang_adapter_config["residual_before_ln"]:
                    residual = hidden_states

                hidden_states, adapter_attention, down, up = self.layer_text_lang_adapters[language](
                    hidden_states, residual_input=residual
                )
                if lang_adapter_config["original_ln_after"]:
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # Task adapters
        # filter tasks that are available in this module
        if tasks:
            tasks = [t for t in tasks if t in self.layer_text_task_adapters]
        if tasks:
            # if we have multiple tasks and use fusion, all configs are assumed to be equal
            task_adapter_config = self.config.adapters.get(tasks[0])
            adapter_used = True

            if task_adapter_config["residual_before_ln"]:
                residual = hidden_states

            if hasattr(self.config, "fusion_config") and self.config.fusion_config["query_before_ln"]:
                query = hidden_states

            if task_adapter_config["original_ln_before"]:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

            if not task_adapter_config["residual_before_ln"]:
                residual = hidden_states

            if hasattr(self.config, "fusion_config") and not self.config.fusion_config["query_before_ln"]:
                query = hidden_states

            # if we have multiple tasks, use fusion
            # TODO ?
            if len(tasks) > 1:
                layer_output_list, down_list, up_list = [], [], []
                # down_list, up_list = [], []
                for task in tasks:
                    intermediate_output, adapter_attention, down, up = self.layer_text_task_adapters[task](
                        hidden_states, residual_input=residual
                    )
                    layer_output_list.append(intermediate_output)
                    # up = self.LayerNorm(intermediate_output )
                    # up = self.LayerNorm(up )
                    down_list.append(down)
                    up_list.append(up)

                layer_output_list = torch.stack(layer_output_list)
                layer_output_list = layer_output_list.permute(1, 2, 0, 3)
                down_list = torch.stack(down_list)
                down_list = down_list.permute(1, 2, 0, 3)
                up_list = torch.stack(up_list)
                up_list = up_list.permute(1, 2, 0, 3)

                attn_name = "_".join(tasks)
                if attn_name not in self.bert_adapter_att:
                    attn_name_new = list(self.bert_adapter_att.keys())[0]
                    # logging.root.warn('{} not in attention layers. Using other attention layer {} instead'.format(
                    #     attn_name,
                    #     attn_name_new
                    # ))
                    attn_name = attn_name_new

                hidden_states = self.bert_adapter_att[attn_name](
                    query, up_list, up_list, residual=residual, attention_mask=attention_mask
                )

                # hidden_states = self.bert_adapter_att[attn_name](query, down_list, up_list, residual=residual, attention_mask=attention_mask)

                # hidden_states += residual

                # hidden_states = up_list[:,:,0] + residual
                # hidden_states = layer_output_list[:,:,0]

                if task_adapter_config["new_attention_norm"]:
                    hidden_states = self.attention_layer_norm(hidden_states + input_tensor)
                else:
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)
            # otherwise, only use one task adapter without attention
            else:
                hidden_states, adapter_attention, down, up = self.layer_text_task_adapters[tasks[0]](
                    hidden_states, residual_input=residual
                )
                if task_adapter_config["original_ln_after"]:
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # In case we haven't used any adapter
        if not adapter_used:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module.
    """

    def add_attention_layer(self, tasks):
        self.attention.output.add_attention_layer(tasks)
        self.output.add_attention_layer(tasks)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        self.attention.output.add_adapter(adapter_name, adapter_type)
        self.output.add_adapter(adapter_name, adapter_type)

    def enable_adapters(self, adapter_type: AdapterType, unfreeze_adapters: bool, unfreeze_attention: bool):
        self.attention.output.enable_adapters(adapter_type, unfreeze_adapters, unfreeze_attention)
        self.output.enable_adapters(adapter_type, unfreeze_adapters, unfreeze_attention)


class BertEncoderAdaptersMixin:
    """Adds adapters to the BertEncoder module.
    """

    def add_attention_layer(self, task_names):
        for layer in self.layer:
            layer.add_attention_layer(task_names)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        if hasattr(adapter_config, 'leave_out'):
            leave_out = adapter_config.leave_out
        else:
            leave_out = []
        for i, layer in enumerate(self.layer):
            if i not in leave_out:
                layer.add_adapter(adapter_name, adapter_type)

    def enable_adapters(self, adapter_type: AdapterType, unfreeze_adapters: bool, unfreeze_attention: bool):
        for layer in self.layer:
            layer.enable_adapters(adapter_type, unfreeze_adapters, unfreeze_attention)


class BertModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the BertModel module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        self.invertible_lang_adapters = nn.ModuleDict(dict())

        # language adapters
        for language in self.config.adapters.adapter_list(AdapterType.text_lang):
            self.encoder.add_adapter(language, AdapterType.text_lang)
            self.add_invertible_lang_adapter(language)
        # task adapters
        for task in self.config.adapters.adapter_list(AdapterType.text_task):
            self.encoder.add_adapter(task, AdapterType.text_task)
        # fusion
        if hasattr(self.config, "fusion_models"):
            for tasks in self.config.fusion_models:
                self.add_attention_layer(tasks)

    def train_adapter(self, adapter_type: AdapterType):
        """Sets the model in mode for training the given type of adapter.
        """
        if not self.has_adapters(adapter_type):
            raise ValueError("No adapters of this type available fro training.")
        self.train()
        self.freeze_model(True)
        self.encoder.enable_adapters(adapter_type, True, False)
        # unfreeze invertible adapters for language adapters
        if adapter_type == AdapterType.text_lang:
            for param in self.invertible_lang_adapters.parameters():
                param.requires_grad = True

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType, config=None):
        """Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            adapter_type (AdapterType): The adapter type.
            config (str or dict, optional): The adapter configuration, can be either:
                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        if not AdapterType.has(adapter_type):
            raise ValueError("Invalid adapter type {}".format(adapter_type))
        if not self.config.adapters.get_config(adapter_type):
            self.config.adapters.set_config(adapter_type, config or DEFAULT_ADAPTER_CONFIG)
        self.config.adapters.add(adapter_name, adapter_type, config=config)
        self.encoder.add_adapter(adapter_name, adapter_type)
        if adapter_type == AdapterType.text_lang:
            self.add_invertible_lang_adapter(adapter_name)

    def add_invertible_lang_adapter(self, language):
        if language in self.invertible_lang_adapters:
            raise ValueError(f"Model already contains an adapter module for '{language}'.")
        inv_adap_config = self.config.adapters.get(language)["invertible_adapter"]
        if inv_adap_config["block_type"] == "nice":
            inv_adap = NICECouplingBlock(
                [[self.config.hidden_size]],
                non_linearity=inv_adap_config["non_linearity"],
                reduction_factor=inv_adap_config["reduction_factor"],
            )
        elif inv_adap_config["block_type"] == "glow":
            inv_adap = GLOWCouplingBlock(
                [[self.config.hidden_size]],
                non_linearity=inv_adap_config["non_linearity"],
                reduction_factor=inv_adap_config["reduction_fector"],
            )
        else:
            raise ValueError(f"Invalid invertible adapter type '{inv_adap_config['block_type']}'.")
        self.invertible_lang_adapters[language] = inv_adap
        self.invertible_lang_adapters[language].apply(Adapter.init_bert_weights)

    def get_invertible_lang_adapter(self, language):
        if language in self.invertible_lang_adapters:
            return self.invertible_lang_adapters[language]
        else:
            return None

    def add_attention_layer(self, task_names):
        """See BertModel.add_attention_layer"""
        self.encoder.add_attention_layer(task_names)


class BertModelHeadsMixin(ModelWithHeadsAdaptersMixin):
    """Adds heads to a Bert-based module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.active_language_adapter = None
        self.active_task_adapters = []
        self.active_head = None

    def _init_head_modules(self):
        self.config.prediction_heads = {}
        self.heads = nn.ModuleDict(dict())
        # add modules for heads in config
        if hasattr(self.config, "prediction_heads"):
            for head_name in self.config.prediction_heads:
                self.add_prediction_head_module(head_name)

    def set_active_language(self, language_name: str):
        """Sets the language adapter which should be used by default in a forward pass.

        Args:
            language_name (str): The name of the language adapter.
        """
        if language_name in self.config.adapters.adapter_list(AdapterType.text_lang):
            self.active_language_adapter = language_name
        else:
            logger.info("No language adapter with name '{}' available.".format(language_name))

    def set_active_task(self, task_name: str):
        """Sets the task adapter and/ or prediction head which should be used by default in a forward pass.
        If no adapter or prediction with the given name is found, no module of the respective type will be activated.

        Args:
            task_name (str): The name of the task adapter and/ or prediction head.
        """
        if task_name in self.config.adapters.adapter_list(AdapterType.text_task):
            self.active_task_adapters = [task_name]
        else:
            logger.info("No task adapter for task_name '{}' available.".format(task_name))
        if task_name in self.config.prediction_heads:
            self.active_head = task_name
        else:
            logger.info("No prediction head for task_name '{}' available.".format(task_name))

    def add_classification_head(
        self, head_name, num_labels=2, layers=2, activation_function="tanh", overwrite_ok=False,
    ):
        """Adds a sequence classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        config = {
            "head_type": "classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_multiple_choice_head(
        self, head_name, num_choices=2, layers=2, activation_function="tanh", overwrite_ok=False,
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
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_tagging_head(
        self, head_name, num_labels=2, layers=1, activation_function="tanh", overwrite_ok=False,
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
        }
        self.add_prediction_head(head_name, config, overwrite_ok)

    def add_prediction_head(
        self, head_name, config, overwrite_ok=False,
    ):
        if head_name not in self.config.prediction_heads or overwrite_ok:
            self.config.prediction_heads[head_name] = config

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

    def forward_head(self, outputs, head_name=None, attention_mask=None, labels=None):
        head_name = head_name or self.active_head
        if not head_name:
            logger.warn("No prediction head is used.")
            return outputs

        if head_name not in self.config.prediction_heads:
            raise ValueError("Unknown head_name '{}'".format(head_name))

        head = self.config.prediction_heads[head_name]

        sequence_output = outputs[0]

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

        elif head["head_type"] == "multiple_choice":
            logits = self.heads[head_name](sequence_output[:, 0])
            logits = logits.view(-1, head["num_choices"])

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                outputs = (loss,) + outputs

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

        else:
            raise ValueError("Unknown head_type '{}'".format(head["head_type"]))

        return outputs  # (loss), logits, (hidden_states), (attentions)
