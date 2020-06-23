import torch
from torch import nn

from .adapter_modeling import *
from .adapter_model_mixin import ModelAdaptersMixin, AdapterType, DEFAULT_ADAPTER_CONFIG


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
        if adapter_config and adapter_config['mh_adapter']:
            adapter = Adapter(input_size=self.config.hidden_size,
                              down_sample=self.config.hidden_size // adapter_config['reduction_factor'],
                              add_layer_norm_before=adapter_config['ln_before'],
                              add_layer_norm_after=adapter_config['ln_after'],
                              non_linearity=adapter_config['non_linearity'],
                              residual_before_ln=adapter_config['adapter_residual_before_ln']
                              )
            if adapter_type == AdapterType.text_task:
                self.attention_text_task_adapters[adapter_name] = adapter
            elif adapter_type == AdapterType.text_lang:
                self.attention_text_lang_adapters[adapter_name] = adapter
            else:
                raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def add_attention_layer(self, tasks):
        """See BertModel.add_attention_layer"""
        task_names = tasks if isinstance(tasks, list) else tasks.split('_')
        adapter_config = self.config.adapters.common_config(task_names)
        if not adapter_config:
            raise ValueError("All tasks used in the attention layer must have the same configuration.")
        if adapter_config['mh_adapter']:
            if adapter_config['attention_type'] == 'tok-lvl':
                layer = BertAdapterAttention(self.config)
            elif adapter_config['attention_type'] == 'sent-lvl':
                layer = AdapterWeightingSentLvl(self.config, len(task_names))
            elif adapter_config['attention_type'] == 'sent-lvl-dynamic':
                layer = AdapterWeightingSentLvlDynamic(self.config, len(task_names))
            elif adapter_config['attention_type'] == 'static':
                layer = SimpleAdapterWeightingStatic(self.config, len(task_names))
            else:
                raise Exception('Unknown attention type: {}'.format(adapter_config['attention_type']))

            self.attention_adapters_fusion['_'.join(task_names)] = layer

            if adapter_config['new_attention_norm']:
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

                if hasattr(self, 'attention_layer_norm'):
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

                if hasattr(self, 'language_attention_layer_norm'):
                    for param in self.language_attention_layer_norm.parameters():
                        param.requires_grad = True
        else:
            raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def adapters_forward(self, hidden_states, input_tensor, tasks=None, language=None):
        adapter_used = False

        # Language adapter
        lang_adapter_config = self.config.adapters.get(language)
        if lang_adapter_config and language in self.attention_text_lang_adapters:
            adapter_used = True

            if lang_adapter_config['residual_before_ln']:
                residual = hidden_states

            if lang_adapter_config['original_ln_before']:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

            if not lang_adapter_config['residual_before_ln']:
                residual = hidden_states

            hidden_states, adapter_attention, down, up = self.attention_text_lang_adapters[language](
                hidden_states,
                residual_input=residual
            )
            if lang_adapter_config['original_ln_after']:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # Task adapters
        # filter tasks that are available in this module
        if tasks:
            tasks = [t for t in tasks if t in self.attention_text_task_adapters]
        if tasks:
            # if we have multiple tasks and use fusion, all configs are assumed to be equal
            task_adapter_config = self.config.adapters.get(tasks[0])
            adapter_used = True

            if task_adapter_config['residual_before_ln']:
                residual = hidden_states

            if hasattr(self.config, 'fusion_config') and self.config.fusion_config['query_before_ln']:
                query = hidden_states

            if task_adapter_config['original_ln_before']:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

            if not task_adapter_config['residual_before_ln']:
                residual = hidden_states

            if hasattr(self.config, 'fusion_config') and not self.config.fusion_config['query_before_ln']:
                query = hidden_states

            # if we have multiple tasks, use fusion
            if len(tasks) > 1:
                # TODO see BertOutput module
                raise NotImplementedError()
                
            # otherwise, only use one task adapter without attention
            else:
                hidden_states, adapter_attention, down, up = self.attention_text_task_adapters[tasks[0]](
                    hidden_states,
                    residual_input=residual
                )
                if task_adapter_config['original_ln_after']:
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
        task_names = tasks if isinstance(tasks, list) else tasks.split('_')
        adapter_config = self.config.adapters.common_config(task_names)
        if not adapter_config:
            raise ValueError("All tasks used in the attention layer must have the same configuration.")
        if adapter_config['output_adapter']:
            if adapter_config['attention_type'] == 'tok-lvl':
                layer = BertAdapterAttention(self.config)
            elif adapter_config['attention_type'] == 'sent-lvl':
                layer = AdapterWeightingSentLvl(self.config, len(task_names))
            elif adapter_config['attention_type'] == 'sent-lvl-dynamic':
                layer = AdapterWeightingSentLvlDynamic(self.config, len(task_names))
            elif adapter_config['attention_type'] == 'static':
                layer = SimpleAdapterWeightingStatic(self.config, len(task_names))
            elif adapter_config['attention_type'] == 'sent-lvl-fusion':
                layer = AdapterFusionSentLvlDynamic(self.config, len(task_names))

            else:
                raise Exception('Unknown attention type: {}'.format(adapter_config['attention_type']))

            self.bert_adapter_att['_'.join(task_names)] = layer

            if adapter_config['new_attention_norm']:
                self.attention_layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

    def add_adapter(self, adapter_name: str, adapter_type: AdapterType):
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config['output_adapter']:
            adapter = Adapter(input_size=self.config.hidden_size,
                              down_sample=self.config.hidden_size // adapter_config['reduction_factor'],
                              add_layer_norm_before=adapter_config['ln_before'],
                              add_layer_norm_after=adapter_config['ln_after'],
                              non_linearity=adapter_config['non_linearity'],
                              residual_before_ln=adapter_config['adapter_residual_before_ln']
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

                if hasattr(self, 'attention_layer_norm'):
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

                if hasattr(self, 'language_attention_layer_norm'):
                    for param in self.language_attention_layer_norm.parameters():
                        param.requires_grad = True
        else:
            raise ValueError("Invalid adapter type '{}'.".format(adapter_type))

    def adapters_forward(self, hidden_states, input_tensor, attention_mask, tasks=None, language=None):
        adapter_used = False

        # Language adapter
        lang_adapter_config = self.config.adapters.get(language)
        if lang_adapter_config and language in self.layer_text_lang_adapters:
            adapter_used = True

            if lang_adapter_config['residual_before_ln']:
                residual = hidden_states

            if lang_adapter_config['original_ln_before']:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

            if not lang_adapter_config['residual_before_ln']:
                residual = hidden_states

            hidden_states, adapter_attention, down, up = self.layer_text_lang_adapters[language](
                hidden_states,
                residual_input=residual
            )
            if lang_adapter_config['original_ln_after']:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # Task adapters
        # filter tasks that are available in this module
        if tasks:
            tasks = [t for t in tasks if t in self.layer_text_task_adapters]
        if tasks:
            # if we have multiple tasks and use fusion, all configs are assumed to be equal
            task_adapter_config = self.config.adapters.get(tasks[0])
            adapter_used = True

            if task_adapter_config['residual_before_ln']:
                residual = hidden_states

            if hasattr(self.config, 'fusion_config') and self.config.fusion_config['query_before_ln']:
                query = hidden_states

            if task_adapter_config['original_ln_before']:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

            if not task_adapter_config['residual_before_ln']:
                residual = hidden_states

            if hasattr(self.config, 'fusion_config') and not self.config.fusion_config['query_before_ln']:
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

                attn_name = '_'.join(tasks)
                if attn_name not in self.bert_adapter_att:
                    attn_name_new = list(self.bert_adapter_att.keys())[0]
                    # logging.root.warn('{} not in attention layers. Using other attention layer {} instead'.format(
                    #     attn_name,
                    #     attn_name_new
                    # ))
                    attn_name = attn_name_new

                hidden_states = self.bert_adapter_att[attn_name](query, up_list, up_list, residual=residual, attention_mask=attention_mask)

                # hidden_states = self.bert_adapter_att[attn_name](query, down_list, up_list, residual=residual, attention_mask=attention_mask)

                # hidden_states += residual

                # hidden_states = up_list[:,:,0] + residual
                # hidden_states = layer_output_list[:,:,0]

                if task_adapter_config['new_attention_norm']:
                    hidden_states = self.attention_layer_norm(hidden_states + input_tensor)
                else:
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)
            # otherwise, only use one task adapter without attention
            else:
                hidden_states, adapter_attention, down, up = self.layer_text_task_adapters[tasks[0]](
                    hidden_states,
                    residual_input=residual
                )
                if task_adapter_config['original_ln_after']:
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
        if adapter_config:
            leave_out = adapter_config.get("leave_out", [])
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
        self.prediction_heads = nn.ModuleDict(dict())

        # language adapters
        for language in self.config.adapters.adapter_list(AdapterType.text_lang):
            self.encoder.add_adapter(language, AdapterType.text_lang)
            self.add_invertible_lang_adapter(language)
        # task adapters
        for task in self.config.adapters.adapter_list(AdapterType.text_task):
            self.encoder.add_adapter(task, AdapterType.text_task)
        # fusion
        if hasattr(self.config, 'fusion_models'):
            for tasks in self.config.fusion_models:
                self.add_attention_layer(tasks)
        # prediction heads
        if hasattr(self.config, 'prediction_heads'):
            for k, v in self.config.prediction_heads.items():
                self.add_prediction_head(task=k,
                                         nr_labels=v['nr_labels'],
                                         task_type=v['task_type'],
                                         layers=v['layers'],
                                         activation_function=v['activation_function'],
                                         qa_examples=v['qa_examples'])

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
        inv_adap_config = self.config.adapters.get(language)['invertible_adapter']
        if inv_adap_config['block_type'] == 'nice':
            inv_adap = NICECouplingBlock(
                [[self.config.hidden_size]],
                non_linearity=inv_adap_config['non_linearity'],
                reduction_factor=inv_adap_config['reduction_factor']
            )
        elif inv_adap_config['block_type'] == 'glow':
            inv_adap = GLOWCouplingBlock(
                [[self.config.hidden_size]],
                non_linearity=inv_adap_config['non_linearity'],
                reduction_factor=inv_adap_config['reduction_fector']
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

    def add_prediction_head(self, task, nr_labels=None, task_type=None, layers=None, activation_function=None, qa_examples=None):
        """Adds a new prediction head to the model.
        """
        if type(task) == str:
            task_name = task
            assert task_type is not None

        else:
            task_name = task['name'].lower()
            task_type = task['task_type'].lower()
            nr_labels = task['data'].get_nr_labels()
            layers = task['nr_layers']
            activation_function = task['activation_function'].lower()
            qa_examples = task['qa_examples']

        if task_type in ['classification', 'tagging']:
            self.__add_classication_head__(task_name=task_name,
                                           nr_labels=nr_labels,
                                           layers=layers,
                                           activation_function=activation_function,
                                           qa_examples=None)
        elif task_type == 'qa':
            self.__add_qa_head__(task_name=task_name,
                                 layers=layers,
                                 activation_function=activation_function,
                                 qa_examples=qa_examples,
                                 nr_labels=None)

        elif task_type == 'extractive_qa':
            # TODO: Check number of labels
            self.__add_squad_head__(task_name=task_name, layers=layers,
                                    activation_function=activation_function,
                                    qa_examples=None,
                                    nr_labels=2)

    def __add_classication_head__(self, task_name, nr_labels, layers, activation_function, qa_examples=None):
        pred_head = []

        for l in range(layers):
            pred_head.append(nn.Dropout(self.config.hidden_dropout_prob))
            if l < layers - 1:
                pred_head.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
                pred_head.append(Activation_Function_Class(activation_function))
            else:
                pred_head.append(nn.Linear(self.config.hidden_size, nr_labels))

        self.prediction_heads[task_name] = nn.Sequential(*pred_head)

        self.prediction_heads[task_name].apply(Adapter.init_bert_weights)

        if not hasattr(self.config, 'prediction_heads'):
            self.config.prediction_heads = {}
        if task_name not in self.config.prediction_heads:
            self.config.prediction_heads[task_name] = {}
            self.config.prediction_heads[task_name]['task_type'] = 'classification'
            self.config.prediction_heads[task_name]['nr_labels'] = nr_labels
            self.config.prediction_heads[task_name]['layers'] = layers
            self.config.prediction_heads[task_name]['activation_function'] = activation_function
            self.config.prediction_heads[task_name]['qa_examples'] = None

    def __add_qa_head__(self, task_name, layers, activation_function, qa_examples, nr_labels=None):

        pred_head = []

        for l in range(layers):
            pred_head.append(nn.Dropout(self.config.hidden_dropout_prob))
            if l < layers - 1:
                pred_head.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
                pred_head.append(Activation_Function_Class(activation_function))
            else:
                pred_head.append(nn.Linear(self.config.hidden_size, 1))

        self.prediction_heads[task_name] = nn.Sequential(*pred_head)

        self.prediction_heads[task_name].apply(Adapter.init_bert_weights)

        if not hasattr(self.config, 'prediction_heads'):
            self.config.prediction_heads = {}
        if task_name not in self.config.prediction_heads:
            self.config.prediction_heads[task_name] = {}
            self.config.prediction_heads[task_name]['task_type'] = 'qa'
            self.config.prediction_heads[task_name]['qa_examples'] = qa_examples
            self.config.prediction_heads[task_name]['layers'] = layers
            self.config.prediction_heads[task_name]['activation_function'] = activation_function
            self.config.prediction_heads[task_name]['nr_labels'] = None

    def __add_squad_head__(self, task_name, layers, activation_function, qa_examples, nr_labels=None):
        pred_head = []

        # for l in range(layers):
        #     pred_head.append(nn.Dropout(self.config.hidden_dropout_prob))
        #     if l < layers - 1:
        #         pred_head.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
        #         pred_head.append(Activation_Function_Class(activation_function))
        #     else:
        #         pred_head.append(nn.Linear(self.config.hidden_size, nr_labels))
        pred_head.append(nn.Linear(self.config.hidden_size, nr_labels))
        self.prediction_heads[task_name] = nn.Sequential(*pred_head)

        self.prediction_heads[task_name].apply(Adapter.init_bert_weights)

        if not hasattr(self.config, 'prediction_heads'):
            self.config.prediction_heads = {}
        if task_name not in self.config.prediction_heads:
            self.config.prediction_heads[task_name] = {}
            self.config.prediction_heads[task_name]['task_type'] = 'extractive_qa'
            self.config.prediction_heads[task_name]['nr_labels'] = 2
            self.config.prediction_heads[task_name]['layers'] = 1
            self.config.prediction_heads[task_name]['activation_function'] = None
            self.config.prediction_heads[task_name]['qa_examples'] = None

    def task_forward(self, task, outputs, sequence_output, valid_ids, device):
        if isinstance(task, str):
            task_name = task
            task = self.config.prediction_heads[task]
        else:
            task_name = task['name']

        if task['task_type'] == 'classification':
            outputs = self.prediction_heads[task_name](outputs[0][:, 0, :])

        elif task['task_type'] == 'qa':
            outputs = self.prediction_heads[task_name](outputs[0][:, 0, :])
            outputs = outputs.view(-1, task['qa_examples'])
            # outputs = outputs.squeeze(-1).view(-1,task['qa_examples'])

        elif task['task_type'] == 'tagging':
            batch_size, max_len, feat_dim = sequence_output.shape
            valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(device)
            for i in range(batch_size):
                jj = -1
                for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
            # sequence_output = self.dropout(valid_output)
            # outputs = self.classifier(sequence_output)
            outputs = self.prediction_heads[task_name](sequence_output)

        elif task['task_type'] == 'extractive_qa':
            sequence_output = outputs[0]
            outputs = self.prediction_heads[task_name](sequence_output)

        return outputs
