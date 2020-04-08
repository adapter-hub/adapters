# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """


import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .activations import gelu, gelu_new, swish
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import PreTrainedModel, prune_linear_layer

from .adapter import *
from .adapters_model import AdaptersModelMixin
from .invertible_lang_adapters import *


logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    "bert-base-japanese": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-pytorch_model.bin",
    "bert-base-japanese-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-pytorch_model.bin",
    "bert-base-japanese-char": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-pytorch_model.bin",
    "bert-base-japanese-char-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
    "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/pytorch_model.bin",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


BertLayerNorm = torch.nn.LayerNorm


class Activation_Function_Class(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self, hidden_act):

        if hidden_act.lower() == 'relu':
            self.f = nn.functional.relu
        elif hidden_act.lower() == 'tanh':
            self.f = torch.tanh
        elif hidden_act.lower() == 'swish':
            def swish(x):
                return x * torch.nn.functional.sigmoid(x)
            self.f = swish
        elif hidden_act.lower() == 'gelu':
            def gelu_new(x):
                """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                    Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            self.f = gelu_new
        elif hidden_act.lower() == 'leakyrelu':
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):

        position_ids = None

        # if self.config.roberta:
        token_type_ids = None
        position_ids = None

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.attention_adapters = nn.ModuleDict(dict())

        self.attention_adapters_fusion = nn.ModuleDict(dict())

        self.adapters_enabled = False
        if hasattr(config, 'adapters'):
            self.adapters_enabled = True
            for task in config.adapters:
               self.add_adapter(task)
            # adaters will be added by BertModel


        self.language_attention_adapters = nn.ModuleDict(dict())

        self.language_attention_adapters_fusion = nn.ModuleDict(dict())

        self.language_adapters_enabled = False
        if hasattr(config, 'language_adapters'):
            self.language_adapters_enabled = True
            for language in config.language_adapters:
               self.add_language_adapter(language)
            # adaters will be added by BertModel



    def add_adapter(self, task_name):
        if self.config.adapter_config['MH_Adapter']:
            self.attention_adapters[task_name] = Adapter(input_size=self.config.hidden_size,
                                                         down_sample=self.config.hidden_size // self.config.adapter_config['reduction_factor'],
                                                         add_layer_norm_before=self.config.adapter_config['LN_before'],
                                                         add_layer_norm_after=self.config.adapter_config['LN_after'],
                                                         non_linearity=self.config.adapter_config['non_linearity'],
                                                         residual_before_ln=self.config.adapter_config[
                                                             'adapter_residual_before_ln']
                                                         )

    def add_attention_layer(self, tasks):
        """See BertModel.add_attention_layer"""
        if self.config.adapter_config['MH_Adapter']:
            task_names = tasks if isinstance(tasks, list) else tasks.split('_')
            if self.config.adapter_config['attention_type'] == 'tok-lvl':
                layer = BertAdapterAttention(self.config)
            else:
                raise Exception('Unknown attention type: {}'.format(self.config.adapter_config['attention_type']))

            self.attention_adapters_fusion['_'.join(task_names)] = layer

            if self.config.adapter_config['new_attention_norm']:
                self.attention_layer_norm = BertLayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

    def enable_adapters(self, unfreeze_adapters, unfreeze_attention):
        if self.config.adapter_config['MH_Adapter']:
            self.adapters_enabled = True
            if unfreeze_adapters:
                for param in self.attention_adapters.parameters():
                    param.requires_grad = True
            if unfreeze_attention:
                for param in self.attention_adapters_fusion.parameters():
                    param.requires_grad = True

                for adap in self.attention_adapters.values():
                    for param in adap.adapter_attention.parameters():
                        param.requires_grad = True

                if self.config.adapter_config['new_attention_norm']:
                    for param in self.attention_layer_norm.parameters():
                        param.requires_grad = True

    def disable_adapters(self):
        self.adapters_enabled = False





    def add_language_adapter(self, task_name):
        if self.config.language_adapter_config['MH_Adapter']:
            self.language_attention_adapters[task_name] = Adapter(input_size=self.config.hidden_size,
                                                         down_sample=self.config.hidden_size // self.config.language_adapter_config['reduction_factor'],
                                                         add_layer_norm_before=self.config.language_adapter_config['LN_before'],
                                                         add_layer_norm_after=self.config.language_adapter_config['LN_after'],
                                                         non_linearity=self.config.language_adapter_config['non_linearity'],
                                                         residual_before_ln=self.config.language_adapter_config[
                                                             'adapter_residual_before_ln']
                                                         )

    def add_language_attention_layer(self, tasks):
        """See BertModel.add_attention_layer"""
        if self.config.language_adapter_config['MH_Adapter']:
            task_names = tasks if isinstance(tasks, list) else tasks.split('_')
            if self.config.language_adapter_config['attention_type'] == 'tok-lvl':
                layer = BertAdapterAttention(self.config)
            else:
                raise Exception('Unknown attention type: {}'.format(self.config.adapter_config['attention_type']))

            self.language_attention_adapters_fusion['_'.join(task_names)] = layer

            if self.config.language_adapter_config['new_attention_norm']:
                self.language_attention_layer_norm = BertLayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

    def enable_language_adapters(self, unfreeze_adapters, unfreeze_attention):
        if self.config.language_adapter_config['MH_Adapter']:
            self.language_adapters_enabled = True
            if unfreeze_adapters:
                for param in self.language_attention_adapters.parameters():
                    param.requires_grad = True
            if unfreeze_attention:
                for param in self.language_attention_adapters_fusion.parameters():
                    param.requires_grad = True

                for adap in self.language_attention_adapters.values():
                    for param in adap.language_adapter_attention.parameters():
                        param.requires_grad = True

                if self.config.language_adapter_config['new_attention_norm']:
                    for param in self.language_attention_layer_norm.parameters():
                        param.requires_grad = True

    def disable_language_adapters(self):
        self.language_adapters_enabled = False

    def forward(self, hidden_states, input_tensor, tasks=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if hasattr(self.config, 'adapter_config') and self.config.adapter_config['MH_Adapter'] and self.adapters_enabled:
            if tasks is None:
                raise Exception('No tasks given, but adapters are active. Deactivate adapters?')

            if self.config.adapter_config['residual_before_ln']:
                residual = hidden_states

            if self.config.adapter_config['original_ln_before']:
                hidden_states = self.LayerNorm(hidden_states + input_tensor)

            if not self.config.adapter_config['residual_before_ln']:
                residual = hidden_states

            if len(tasks) > 1:
                # we use adapter attention
                layer_output_list, down_list = [], []
                for task in tasks:
                    intermediate_output, adapter_attention, down = self.layer_adapters[task](
                        hidden_states, residual_input=residual
                    )
                    layer_output_list.append(intermediate_output)
                    down_list.append(down)

                layer_output_list = torch.stack(layer_output_list)
                layer_output_list = layer_output_list.permute(1, 2, 0, 3)
                down_list = torch.stack(down_list)
                down_list = down_list.permute(1, 2, 0, 3)

                attn_name = '_'.join(tasks)
                if attn_name not in self.bert_adapter_att:
                    attn_name_new = list(self.bert_adapter_att.keys())[0]
                    # logging.root.warn('{} not in attention layers. Using other attention layer {} instead'.format(
                    #     attn_name,
                    #     attn_name_new
                    # ))
                    attn_name = attn_name_new

                hidden_states = self.bert_adapter_att[attn_name](hidden_states, down_list, layer_output_list, attention_mask)



                if self.config.adapter_config['new_attention_norm']:
                    hidden_states = self.attention_layer_norm(hidden_states + input_tensor)
                else:
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)
            else:
                # we use only one task adapter without attention
                hidden_states, adapter_attention, down, up = self.attention_adapters[tasks[0]](
                    hidden_states,
                    residual_input=residual
                )
                if self.config.adapter_config['original_ln_after']:
                    hidden_states = self.LayerNorm(hidden_states + input_tensor)

                # hidden_states = self.LayerNorm(hidden_states + input_tensor)

        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        tasks=None
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states, tasks=tasks)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        # self.bert_adapter_att = BertAdapterAttention(config)
        # self.bert_adapter_att = SimpleAdapterWeightingSentLvl(config)
        self.bert_adapter_att = nn.ModuleDict(dict())

        self.layer_adapters = nn.ModuleDict(dict())

        if hasattr(self.config, 'adapters'):
            self.adapters_enabled = True

            for task in config.adapters:
               self.add_adapter(task)
        else:
            self.adapters_enabled = False


        self.bert_language_adapter_att = nn.ModuleDict(dict())

        self.layer_language_adapters = nn.ModuleDict(dict())

        if hasattr(self.config, 'language_adapters'):
            self.language_adapters_enabled = True

            for language in config.language_adapters:
               self.add_language_adapter(language)
        else:
            self.language_adapters_enabled = False

            # adapters will be added by BertModel

        if hasattr(config, 'fusion_models'):
           for tasks in config.fusion_models:
               self.add_attention_layer(tasks)

    def add_attention_layer(self, tasks):
        """See BertModel.add_attention_layer"""
        if self.config.adapter_config['Output_Adapter']:
            task_names = tasks if isinstance(tasks, list) or isinstance(tasks, tuple) else tasks.split('_')
            if self.config.adapter_config['attention_type'] == 'tok-lvl':
                layer = BertAdapterAttention(self.config)

            else:
                raise Exception('Unknown attention type: {}'.format(self.config.adapter_config['attention_type']))

            self.bert_adapter_att['_'.join(task_names)] = layer

            if self.config.adapter_config['new_attention_norm']:
                self.attention_layer_norm = BertLayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

    def add_adapter(self, task_name):
        if self.config.adapter_config['Output_Adapter']:
            self.layer_adapters[task_name] = Adapter(input_size=self.config.hidden_size,
                                                         down_sample=self.config.hidden_size // self.config.adapter_config['reduction_factor'],
                                                         add_layer_norm_before=self.config.adapter_config['LN_before'],
                                                         add_layer_norm_after=self.config.adapter_config['LN_after'],
                                                         non_linearity=self.config.adapter_config['non_linearity'],
                                                     residual_before_ln=self.config.adapter_config['adapter_residual_before_ln']
                                                     )

    def disable_adapters(self):
        self.adapters_enabled = False

    def enable_adapters(self, unfreeze_adapters, unfreeze_attention):
        if self.config.adapter_config['Output_Adapter']:
            self.adapters_enabled = True
            if unfreeze_adapters:
                for param in self.layer_adapters.parameters():
                    param.requires_grad = True
            if unfreeze_attention:
                for adap in self.layer_adapters.values():
                    for param in adap.adapter_attention.parameters():
                        param.requires_grad = True

                for param in self.bert_adapter_att.parameters():
                    param.requires_grad = True

                if self.config.adapter_config['new_attention_norm']:
                    for param in self.attention_layer_norm.parameters():
                        param.requires_grad = True




    def add_language_adapter(self, task_name):
        if self.config.language_adapter_config['Output_Adapter']:
            self.layer_language_adapters[task_name] = Adapter(input_size=self.config.hidden_size,
                                                         down_sample=self.config.hidden_size // self.config.language_adapter_config['reduction_factor'],
                                                         add_layer_norm_before=self.config.language_adapter_config['LN_before'],
                                                         add_layer_norm_after=self.config.language_adapter_config['LN_after'],
                                                         non_linearity=self.config.language_adapter_config['non_linearity'],
                                                     residual_before_ln=self.config.language_adapter_config['adapter_residual_before_ln']
                                                     )

    def disable_language_adapters(self):
        self.language_adapters_enabled = False

    def enable_language_adapters(self, unfreeze_adapters, unfreeze_attention):
        if self.config.language_adapter_config['Output_Adapter']:
            self.language_adapters_enabled = True
            if unfreeze_adapters:
                for param in self.layer_language_adapters.parameters():
                    param.requires_grad = True
            if unfreeze_attention:
                for adap in self.layer_language_adapters.values():
                    for param in adap.adapter_attention.parameters():
                        param.requires_grad = True

                for param in self.bert_language_adapter_att.parameters():
                    param.requires_grad = True

                if self.config.language_adapter_config['new_attention_norm']:
                    for param in self.language_attention_layer_norm.parameters():
                        param.requires_grad = True


    def forward(self, hidden_states, input_tensor, attention_mask, tasks=None, language=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        used_adapter = False

        if hasattr(self.config, 'language_adapter_config') and self.config.language_adapter_config['Output_Adapter'] and self.language_adapters_enabled :
            # print('trining language adapter')
            if language is None:
                raise Exception('No language given, but adapters are active. Deactivate adapters?')

            # if self.config.language_adapter_config['residual_before_ln']:
            #     residual = hidden_states * 1.0

            # if self.config.language_adapter_config['original_ln_before']:
            # hidden_states = self.LayerNorm(hidden_states + input_tensor)

            # if not self.config.language_adapter_config['residual_before_ln']:
            # residual = hidden_states * 1.0

            # else:
            # we use only one task adapter without attention
            hidden_states, adapter_attention, down, up = self.layer_language_adapters[language](
                hidden_states,
                residual_input=residual
            )
            residual = hidden_states
            # if self.config.language_adapter_config['original_ln_after']:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

            used_adapter = True


        if hasattr(self.config, 'adapter_config')  and self.config.adapter_config['Output_Adapter'] and self.adapters_enabled:
            if tasks is None:
                raise Exception('No tasks given, but adapters are active. Deactivate adapters?')

            # if self.config.adapter_config['residual_before_ln']:
            #     residual = hidden_states * 1.0
            #
            # if hasattr(self.config, 'fusion_config') and self.config.fusion_config['query_before_ln']:
            #     query = hidden_states * 1.0

            # if self.config.adapter_config['original_ln_before']:
            #     hidden_states = self.LayerNorm(hidden_states + input_tensor)

            # if not self.config.adapter_config['residual_before_ln']:
            # residual = hidden_states * 1.0

            # if  hasattr(self.config, 'fusion_config') and not self.config.fusion_config['query_before_ln']:
            #     query = hidden_states * 1.0

            # if len(tasks) > 1:
            #     # we use adapter attention
            #     layer_output_list, down_list, up_list = [], [], []
            #     # down_list, up_list = [], []
            #     for task in tasks:
            #         intermediate_output, adapter_attention, down, up = self.layer_adapters[task](
            #             hidden_states, residual_input=residual
            #         )
            #         layer_output_list.append(intermediate_output)
            #
            #         # up = self.LayerNorm(intermediate_output )
            #         # up = self.LayerNorm(up )
            #
            #         down_list.append(down)
            #         up_list.append(up)
            #
            #     layer_output_list = torch.stack(layer_output_list)
            #     layer_output_list = layer_output_list.permute(1, 2, 0, 3)
            #     down_list = torch.stack(down_list)
            #     down_list = down_list.permute(1, 2, 0, 3)
            #     up_list = torch.stack(up_list)
            #     up_list = up_list.permute(1, 2, 0, 3)
            #
            #     attn_name = '_'.join(tasks)
            #     if attn_name not in self.bert_adapter_att:
            #         attn_name_new = list(self.bert_adapter_att.keys())[0]
            #         # logging.root.warn('{} not in attention layers. Using other attention layer {} instead'.format(
            #         #     attn_name,
            #         #     attn_name_new
            #         # ))
            #         attn_name = attn_name_new
            #
            #     hidden_states = self.bert_adapter_att[attn_name](query, up_list, up_list, residual=residual, attention_mask=attention_mask)
            #
            #     # hidden_states = self.bert_adapter_att[attn_name](query, down_list, up_list, residual=residual, attention_mask=attention_mask)
            #
            #     # hidden_states += residual
            #
            #     # hidden_states = up_list[:,:,0] + residual
            #     # hidden_states = layer_output_list[:,:,0]
            #
            #     if self.config.adapter_config['new_attention_norm']:
            #         hidden_states = self.attention_layer_norm(hidden_states + input_tensor)
            #     else:
            #         hidden_states = self.LayerNorm(hidden_states + input_tensor)
            # else:
            # we use only one task adapter without attention
            hidden_states, adapter_attention, down, up = self.layer_adapters[tasks[0]](
                hidden_states,
                residual_input=residual
            )
            # if self.config.adapter_config['original_ln_after']:
            # hidden_states = self.LayerNorm(hidden_states + input_tensor)

            used_adapter = True
            hidden_states = self.LayerNorm(hidden_states + input_tensor)


        # if used_adapter:
        #     hidden_states = self.LayerNorm(hidden_states + input_tensor)


            # hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # if not self.config.language_adapter_config['Output_Adapter'] and not (hasattr(self.config, 'adapter_config')  and self.config.adapter_config['Output_Adapter']):
        #     hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = False
        # if self.is_decoder:
        #     self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def add_attention_layer(self, tasks):
        self.attention.output.add_attention_layer(tasks)
        self.output.add_attention_layer(tasks)

    def add_adapter(self, task_name):
        self.attention.output.add_adapter(task_name)
        self.output.add_adapter(task_name)

    def disable_adapters(self):
        self.attention.output.disable_adapters()
        self.output.disable_adapters()

    def enable_adapters(self, unfreeze_adapters, unfreeze_attention):
        self.attention.output.enable_adapters(unfreeze_adapters, unfreeze_attention)
        self.output.enable_adapters(unfreeze_adapters, unfreeze_attention)


    def add_language_adapter(self, task_name):
        self.attention.output.add_language_adapter(task_name)
        self.output.add_language_adapter(task_name)

    def disable_language_adapters(self):
        self.attention.output.disable_language_adapters()
        self.output.disable_language_adapters()

    def enable_language_adapters(self, unfreeze_adapters, unfreeze_attention):
        self.attention.output.enable_language_adapters(unfreeze_adapters, unfreeze_attention)
        self.output.enable_language_adapters(unfreeze_adapters, unfreeze_attention)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        tasks=None, language=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, tasks=tasks,)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, attention_mask, tasks=tasks, language=language)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def add_attention_layer(self, task_names):
        for layer in self.layer:
            layer.add_attention_layer(task_names)

    def add_adapter(self, task_name):
        for layer in self.layer:
            layer.add_adapter(task_name)

    def disable_adapters(self):
        for layer in self.layer:
            layer.disable_adapters()

    def enable_adapters(self, unfreeze_adapters, unfreeze_attention):
        for layer in self.layer:
            layer.enable_adapters(unfreeze_adapters, unfreeze_attention)

    def add_language_adapter(self, task_name):
        for layer in self.layer:
            layer.add_language_adapter(task_name)

    def disable_language_adapters(self):
        for layer in self.layer:
            layer.disable_language_adapters()

    def enable_language_adapters(self, unfreeze_adapters, unfreeze_attention):
        for layer in self.layer:
            layer.enable_language_adapters(unfreeze_adapters, unfreeze_attention)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        tasks=None, language=None
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask,
                tasks=tasks, language=language
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel, AdaptersModelMixin):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.prediction_heads = nn.ModuleDict(dict())

        self.inv_lang_adap = None
        if hasattr(self.config, 'language_adapters'):
            for language in self.config.language_adapters:
                self.add_model_inv_lang_adapter(language)

        if hasattr(self.config, 'prediction_heads'):
            for k,v in self.config.prediction_heads.items():
                self.add_prediction_head(task=k,
                                         nr_labels=v['nr_labels'],
                                         task_type=v['task_type'],
                                         layers=v['layers'],
                                         activation_function=v['activation_function'],
                                         qa_examples=v['qa_examples'])

        self.init_weights()

    def add_model_inv_lang_adapter(self, language):
        if not self.inv_lang_adap:
            self.inv_lang_adap = nn.ModuleDict(dict())
        if language not in self.inv_lang_adap:
            self.inv_lang_adap[language] = NICECouplingBlock([[768]])
            self.inv_lang_adap[language].apply(Adapter.init_bert_weights)


    def add_prediction_head(self, task, nr_labels=None, task_type=None, layers=None, activation_function=None, qa_examples=None):
        # if task['name'] not in self.prediction_heads:

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
            #TODO: Check number of labels
            self.__add_squad_head__(task_name=task_name, layers=layers,
                                    activation_function=activation_function,
                                    qa_examples=None,
                                    nr_labels=2)


    def __add_classication_head__(self, task_name, nr_labels, layers, activation_function, qa_examples=None):
        pred_head = []

        for l in range(layers):
            pred_head.append(nn.Dropout(self.config.hidden_dropout_prob))
            if l < layers-1:
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
            if l < layers-1:
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

    def add_adapter(self, task_name ):
        """See BertModel.add_adapter

        :param type: either bapna or houlsby
        """
        self.encoder.add_adapter(task_name)

    def add_language_adapter(self, language_name ):
        """See BertModel.add_adapter

        :param type: either bapna or houlsby
        """
        self.encoder.add_language_adapter(language_name)

    def add_attention_layer(self, task_names):
        """See BertModel.add_attention_layer"""
        self.encoder.add_attention_layer(task_names)

    def enable_adapters(self, unfreeze_adapters=True, unfreeze_attention=True):
        """Enables the use of adapters. If not enabled, adapters won't be called (default: disabled)

        :param unfreeze_adapters: if set to true, will set requires_grad to true for all params of the adapter
                                    (including the attention)
        :param unfreeze_attention: if set to true, will set requires_grad to the adapter attention parameters
        """
        self.encoder.enable_adapters(unfreeze_adapters, unfreeze_attention)

    def disable_adapters(self):
        """Disables adapters. The model will behave just like a regular BERT model without adapters"""
        self.encoder.disable_adapters()

    def enable_language_adapters(self, unfreeze_adapters=True, unfreeze_attention=True):
        """Enables the use of adapters. If not enabled, adapters won't be called (default: disabled)

        :param unfreeze_adapters: if set to true, will set requires_grad to true for all params of the adapter
                                    (including the attention)
        :param unfreeze_attention: if set to true, will set requires_grad to the adapter attention parameters
        """
        self.encoder.enable_language_adapters(unfreeze_adapters, unfreeze_attention)

    def disable_language_adapters(self):
        """Disables adapters. The model will behave just like a regular BERT model without adapters"""
        self.encoder.disable_language_adapters()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        task=None,
        adapter_tasks=None,
        valid_ids=None,
        language=None,
        inv_lang_adap=None
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        # if self.config.roberta:
        token_type_ids = None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            # if self.config.is_decoder:
            #     batch_size, seq_length = input_shape
            #     seq_ids = torch.arange(seq_length, device=device)
            #     causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            #     causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
            #     extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            # else:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.is_decoder and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        #
        #     if encoder_attention_mask.dim() == 3:
        #         encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        #     elif encoder_attention_mask.dim() == 2:
        #         encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        #     else:
        #         raise ValueError("Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(encoder_hidden_shape,
        #                                                                                                                        encoder_attention_mask.shape))
        #
        #     encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        #     encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        # else:
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if inv_lang_adap:
            embedding_output = inv_lang_adap(embedding_output, rev=False)
        elif self.inv_lang_adap:
            embedding_output = self.inv_lang_adap[language](embedding_output, rev=False)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            tasks=adapter_tasks,
            language=language
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here

        if task:

            if task['task_type'] == 'classification':
                outputs = self.prediction_heads[task['name']](outputs[0][:,0,:])

            elif task['task_type'] == 'qa':
                outputs = self.prediction_heads[task['name']](outputs[0][:,0,:])
                outputs = outputs.view(-1,task['qa_examples'])
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
                outputs = self.prediction_heads[task['name']](sequence_output)

            elif task['task_type'] == 'extractive_qa':
                sequence_output = outputs[0]
                outputs = self.prediction_heads[task['name']](sequence_output)


        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
    a `next sentence prediction (classification)` head. """,
    BERT_START_DOCSTRING,
)
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        next_sentence_label=None,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False
            continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.


    Examples::

        from transformers import BertTokenizer, BertForPreTraining
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        prediction_scores, seq_relationship_scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                Next token prediction loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import BertTokenizer, BertForMaskedLM
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss, prediction_scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top. """, BERT_START_DOCSTRING,
)
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
    ):
        r"""
        next_sentence_label (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`next_sentence_label` is provided):
            Next sequence prediction (classification) loss.
        seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForNextSentencePrediction
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        seq_relationship_scores = outputs[0]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    BERT_START_DOCSTRING,
)
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForMultipleChoice
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]

        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, classification_scores = outputs[:2]

        """
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    BERT_START_DOCSTRING,
)
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
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

        return outputs  # (loss), scores, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`). """,
    BERT_START_DOCSTRING,
)
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForQuestionAnswering
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        assert answer == "a nice puppet"

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
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

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
