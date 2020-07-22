import logging

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .adapter_config import DEFAULT_ADAPTER_CONFIG, AdapterType
from .adapter_model_mixin import ModelAdaptersMixin, ModelWithHeadsAdaptersMixin
from .adapter_modeling import Activation_Function_Class, Adapter, BertFusion, GLOWCouplingBlock, NICECouplingBlock
from .adapter_utils import parse_adapter_names


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


class BertSelfOutputAdaptersMixin:
    """Adds adapters to the BertSelfOutput module.
    """

    def _init_adapter_modules(self):
        self.attention_text_task_adapters = nn.ModuleDict(dict())
        self.adapter_fusion_layer = nn.ModuleDict(dict())
        self.attention_text_lang_adapters = nn.ModuleDict(dict())

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

    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_attention_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        adapter_config = self.config.adapters.common_config(adapter_names)
        if not adapter_config:
            raise ValueError("All tasks used in the attention layer must have the same configuration.")
        if adapter_config["mh_adapter"]:
            self.adapter_fusion_layer[",".join(adapter_names)] = BertFusion(self.config)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_fusion: bool):
        """Unfreezes a given list of adapters, the adapter fusion layer, or both

        :param adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
        :param unfreeze_adapters: whether the adapters themselves should be unfreezed
        :param unfreeze_fusion: whether the adapter attention layer for the given adapters should be unfreezed
        """
        if unfreeze_adapters:
            if isinstance(adapter_names, str):
                adapter_names = [adapter_names]
            for adapter_name in adapter_names:
                layer = self.get_adapter_layer(adapter_name)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_names[0], str):
                adapter_names = [adapter_names]
            for adapter_fusion_group in adapter_names:
                fusion_name = ",".join(adapter_fusion_group)
                if fusion_name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[fusion_name].parameters():
                        param.requires_grad = True

    def get_adapter_preparams(
        self, adapter_config, hidden_states, input_tensor,
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
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        if not adapter_config["residual_before_ln"]:
            residual = hidden_states

        if hasattr(self.config, "adapter_fusion") and not self.config.adapter_fusion["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def get_adapter_layer(self, adapter_name):
        """
        Depending on the adapter type we retrieve the correct layer. If no adapter for that name was set at that layer
        we return None
        Args:
            adapter_name: string name of the adapter

        Returns: layer | None

        """
        if adapter_name in self.attention_text_lang_adapters:
            return self.attention_text_lang_adapters[adapter_name]
        if adapter_name in self.attention_text_task_adapters:
            return self.attention_text_task_adapters[adapter_name]
        return None

    def adapter_stack_layer(self, hidden_states, input_tensor, attention_mask, adapter_stack):
        """
        One layer of stacked adapters. This either passes through a single adapter and prepares the data to be passed
        into a subsequent adapter, or the next transformer layer
        OR
        IFF more than one adapter names is set for one stack layer, we assume that fusion is activated. Thus, the
        adapters are fused together.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            input_tensor: residual connection of transformer
            attention_mask: attention mask on token level
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters

        Returns: hidden_states

        """
        # We assume that all adapters have the same residual connection and layer norm setting as the first adapter in
        # the stack
        adapter_config = self.config.adapters.get(adapter_stack[0])

        hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        if len(adapter_stack) == 1:

            adapter_layer = self.get_adapter_layer(adapter_stack[0])
            if adapter_layer is not None:
                hidden_states, _, _ = adapter_layer(hidden_states, residual_input=residual)

            return hidden_states

        else:
            return self.adapter_fusion(hidden_states, attention_mask, adapter_stack, residual, query)

    def adapter_fusion(self, hidden_states, attention_mask, adapter_stack, residual, query):
        """
        If more than one adapter name is set for a stack layer, we fuse the adapters.
        For this, we pass through every adapter and learn an attention-like weighting of each adapter.
        The information stored in each of the adapters is thus fused together wrt the current example.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            attention_mask: attention mask on token level
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters
            residual: residual of the previous layer
            query: query by which we attend over the adapters

        Returns: hidden_states

        """

        up_list = []

        for adapter_name in adapter_stack:
            adapter_layer = self.get_adapter_layer(adapter_name)
            if adapter_layer is not None:
                intermediate_output, _, up = adapter_layer(hidden_states, residual_input=residual)
                up_list.append(up)
        if len(up_list) > 0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)

            fusion_name = ",".join(adapter_stack)

            hidden_states = self.adapter_fusion_layer[fusion_name](
                query, up_list, up_list, residual=residual, attention_mask=attention_mask
            )
        return hidden_states

    def adapters_forward(self, hidden_states, input_tensor, attention_mask, adapter_names=None):

        if adapter_names is not None:
            adapter_names = parse_adapter_names(adapter_names)
            flat_adapter_names = [item for sublist in adapter_names for item in sublist]

        if adapter_names is not None and (
            len(
                (set(self.attention_text_task_adapters.keys()) | set(self.attention_text_lang_adapters.keys()))
                & set(flat_adapter_names)
            )
            > 0
        ):

            for adapter_stack in adapter_names:
                hidden_states = self.adapter_stack_layer(
                    hidden_states=hidden_states,
                    input_tensor=input_tensor,
                    attention_mask=attention_mask,
                    adapter_stack=adapter_stack,
                )

            last_config = self.config.adapters.get(adapter_names[-1][-1])
            if last_config["original_ln_after"]:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertOutputAdaptersMixin:
    """Adds adapters to the BertOutput module.
    """

    def _init_adapter_modules(self):
        # self.bert_adapter_att = BertAdapterAttention(config)
        # self.bert_adapter_att = SimpleAdapterWeightingSentLvl(config)
        self.adapter_fusion_layer = nn.ModuleDict(dict())
        self.layer_text_task_adapters = nn.ModuleDict(dict())
        self.layer_text_lang_adapters = nn.ModuleDict(dict())

    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_fusion_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        adapter_config = self.config.adapters.common_config(adapter_names)
        if not adapter_config:
            raise ValueError("All tasks used in the fusion layer must have the same configuration.")
        if adapter_config["output_adapter"]:
            self.adapter_fusion_layer[",".join(adapter_names)] = BertFusion(self.config)

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

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_fusion: bool):

        if unfreeze_adapters:
            if isinstance(adapter_names, str):
                adapter_names = [adapter_names]
            for adapter_name in adapter_names:
                layer = self.get_adapter_layer(adapter_name)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_names[0], str):
                adapter_names = [adapter_names]
            for adapter_fusion_group in adapter_names:
                fusion_name = ",".join(adapter_fusion_group)
                if fusion_name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[fusion_name].parameters():
                        param.requires_grad = True

    def get_adapter_preparams(
        self, adapter_config, hidden_states, input_tensor,
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
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        if not adapter_config["residual_before_ln"]:
            residual = hidden_states

        if hasattr(self.config, "adapter_fusion") and not self.config.adapter_fusion["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def get_adapter_layer(self, adapter_name):
        """
        Depending on the adapter type we retrieve the correct layer. If no adapter for that name was set at that layer
        we return None
        Args:
            adapter_name: string name of the adapter

        Returns: layer | None

        """
        if adapter_name in self.layer_text_lang_adapters:
            return self.layer_text_lang_adapters[adapter_name]
        if adapter_name in self.layer_text_task_adapters:
            return self.layer_text_task_adapters[adapter_name]
        return None

    def adapter_stack_layer(self, hidden_states, input_tensor, attention_mask, adapter_stack):
        """
        One layer of stacked adapters. This either passes through a single adapter and prepares the data to be passed
        into a subsequent adapter, or the next transformer layer
        OR
        IFF more than one adapter names is set for one stack layer, we assume that fusion is activated. Thus, the
        adapters are fused together.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            input_tensor: residual connection of transformer
            attention_mask: attention mask on token level
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters

        Returns: hidden_states

        """
        # We assume that all adapters have the same residual connection and layer norm setting as the first adapter in
        # the stack
        adapter_config = self.config.adapters.get(adapter_stack[0])

        hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)

        if len(adapter_stack) == 1:

            adapter_layer = self.get_adapter_layer(adapter_stack[0])
            if adapter_layer is not None:
                hidden_states, _, _ = adapter_layer(hidden_states, residual_input=residual)

            return hidden_states

        else:
            return self.adapter_fusion(hidden_states, attention_mask, adapter_stack, residual, query)

    def adapter_fusion(self, hidden_states, attention_mask, adapter_stack, residual, query):
        """
        If more than one adapter name is set for a stack layer, we fuse the adapters.
        For this, we pass through every adapter and learn an attention-like weighting of each adapter.
        The information stored in each of the adapters is thus fused together wrt the current example.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            attention_mask: attention mask on token level
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters
            residual: residual of the previous layer
            query: query by which we attend over the adapters

        Returns: hidden_states

        """
        up_list = []

        for adapter_name in adapter_stack:
            adapter_layer = self.get_adapter_layer(adapter_name)
            if adapter_layer is not None:
                intermediate_output, _, up = adapter_layer(hidden_states, residual_input=residual)
                up_list.append(up)

        if len(up_list) > 0:

            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)

            fusion_name = ",".join(adapter_stack)

            hidden_states = self.adapter_fusion_layer[fusion_name](
                query, up_list, up_list, residual=residual, attention_mask=attention_mask
            )
        return hidden_states

    def adapters_forward(self, hidden_states, input_tensor, attention_mask, adapter_names=None):

        if adapter_names is not None:
            adapter_names = parse_adapter_names(adapter_names)

            flat_adapter_names = [item for sublist in adapter_names for item in sublist]

        if adapter_names is not None and (
            len(
                (set(self.layer_text_lang_adapters.keys()) | set(self.layer_text_task_adapters.keys()))
                & set(flat_adapter_names)
            )
            > 0
        ):

            for adapter_stack in adapter_names:
                hidden_states = self.adapter_stack_layer(
                    hidden_states=hidden_states,
                    input_tensor=input_tensor,
                    attention_mask=attention_mask,
                    adapter_stack=adapter_stack,
                )

            last_config = self.config.adapters.get(adapter_names[-1][-1])
            if last_config["original_ln_after"]:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module.
    """

    def add_fusion_layer(self, adapter_names):
        self.attention.output.add_fusion_layer(adapter_names)
        self.output.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        self.attention.output.add_adapter(adapter_name, adapter_type)
        self.output.add_adapter(adapter_name, adapter_type)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        self.attention.output.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)
        self.output.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)


class BertEncoderAdaptersMixin:
    """Adds adapters to the BertEncoder module.
    """

    def add_fusion_layer(self, adapter_names):
        for layer in self.layer:
            layer.add_fusion_layer(adapter_names)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        if hasattr(adapter_config, "leave_out"):
            leave_out = adapter_config.leave_out
        else:
            leave_out = []
        for i, layer in enumerate(self.layer):
            if i not in leave_out:
                layer.add_adapter(adapter_name, adapter_type)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool, unfreeze_attention: bool):
        for layer in self.layer:
            layer.enable_adapters(adapter_names, unfreeze_adapters, unfreeze_attention)


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
            for fusion_adapter_names in self.config.fusion_models:
                self.add_fusion_layer(fusion_adapter_names)

    def train_adapter(self, adapter_names: list):
        """Sets the model in mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        self.encoder.enable_adapters(adapter_names, True, False)
        # unfreeze invertible adapters for invertible adapters
        for adapter_name in adapter_names:
            if adapter_name in self.invertible_lang_adapters:
                for param in self.invertible_lang_adapters[adapter_name].parameters():
                    param.requires_grad = True

    def train_fusion(self, adapter_names: list):
        """Sets the model in mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        self.encoder.enable_adapters(adapter_names, False, True)
        # TODO implement fusion for invertible adapters

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType, config=None):
        """Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            adapter_type (AdapterType): The adapter type.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:
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
                reduction_factor=inv_adap_config["reduction_factor"],
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

    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_attention_layer"""
        self.encoder.add_fusion_layer(adapter_names)


class BertModelHeadsMixin(ModelWithHeadsAdaptersMixin):
    """Adds heads to a Bert-based module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.active_adapter_names = None
        self.active_head = None

    def _init_head_modules(self):
        if not hasattr(self.config, "prediction_heads"):
            self.config.prediction_heads = {}
        self.heads = nn.ModuleDict(dict())
        # add modules for heads in config
        for head_name in self.config.prediction_heads:
            self._add_prediction_head_module(head_name)

    def set_active_adapters(self, adapter_names: list):
        """Sets the task adapter and/ or prediction head which should be used by default in a forward pass.
        If no adapter or prediction with the given name is found, no module of the respective type will be activated.

        Args:
            task_name (str): The name of the task adapter and/ or prediction head.
        """
        adapter_names = parse_adapter_names(adapter_names)

        new_adapter_names = []

        for stack in adapter_names:
            new_adapter_names.append([])
            for adapter_name in stack:
                if adapter_name in self.config.adapters.adapter_list(
                    AdapterType.text_task
                ) or adapter_name in self.config.adapters.adapter_list(AdapterType.text_lang):
                    new_adapter_names[-1].append(adapter_name)
                else:
                    logger.info("No task adapter for task_name '{}' available. Removing it.".format(adapter_name))
                if adapter_name in self.config.prediction_heads:
                    self.active_head = adapter_name
                else:
                    logger.info("No prediction head for task_name '{}' available.".format(adapter_name))
        if len(new_adapter_names[0]) == 0:
            new_adapter_names = None
        self.active_adapter_names = new_adapter_names

    def add_classification_head(
        self, head_name, num_labels=2, layers=2, activation_function="tanh", overwrite_ok=False, multilabel=False
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

        elif head["head_type"] == "multilabel_classification":
            logits = self.heads[head_name](sequence_output[:, 0])

            outputs = (logits,) + outputs[2:]
            if labels is not None:
                loss_fct = BCEWithLogitsLoss()
                if labels.dtype != torch.float32:
                    labels = labels.float()
                loss = loss_fct(logits, labels)
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
