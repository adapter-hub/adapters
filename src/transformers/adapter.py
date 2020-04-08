import torch
from torch import nn
import math
# TODO not sure if this is right...
from torch.nn import LayerNorm as BertLayerNorm

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


class Adapter(nn.Module):
    def __init__(self, input_size, down_sample=None, non_linearity = 'relu', init_bert_weights=True,
                 add_layer_norm_before=True, add_layer_norm_after=False, residual_before_ln=True):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = BertLayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        # Linear down projection of the input
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        # TODO give more options than just relu, or pass the non_linearity directly, not as a string
        # if non_linearity.lower() == 'relu':
        #     self.non_linearity = nn.ReLU()
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # attention layer that learns the usefulness of the different adapters. Is only trained in the later steps
        self.adapter_attention = nn.Linear(self.down_sample, 1)

        # Up projection to input size
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = BertLayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if init_bert_weights:
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_attention.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)

    def forward(self, x, residual_input): #, residual_input=None):
        down = self.adapter_down(x)
        attention = self.adapter_attention(down)
        up = self.adapter_up(down)

        # output = x + up
        # if residual_input is not None:
        #     output += residual_input

        output = up

        if self.residual_before_ln:
            output += residual_input

        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        if not self.residual_before_ln:
            output += residual_input

        return output, attention, down, up

    # This is copied from the BERT model so that this is a self containing class. This unfortunately introduces code
    # copying so it might be better to pass the BERT model here TODO
    @staticmethod
    def init_bert_weights(module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # TODO I set the std to default 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertAdapterAttention(nn.Module):
    def __init__(self, config):
        super(BertAdapterAttention, self).__init__()
        # if config.hidden_size % config.num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.config = config
        self.output_attentions = config.output_attentions

        self.dense_size = int(config.hidden_size) // config.adapter_config['reduction_factor']

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.T = 50.0

        self.reduction = self.T / 1000.0

        if not self.config.fusion_config['query'] and \
                not self.config.fusion_config['key'] and \
                not self.config.fusion_config['value']:
            self.dense = nn.Linear(self.dense_size, 1)

        if self.config.fusion_config['query'] :
            self.query = nn.Linear(int(config.hidden_size) , self.dense_size)

        if self.config.fusion_config['key'] :
            self.key = nn.Linear(self.dense_size, self.dense_size)

        if self.config.fusion_config['value'] :
            self.value =  nn.Linear(int(config.hidden_size), int(config.hidden_size))

        if self.config.fusion_config['temperature']:
            self.T = 50.0
        else:
            self.T = 1.0

        self.reduction = self.T / 1000.0

    def forward(self, query, key, value, residual, attention_mask=None):

        if self.config.fusion_config['residual_before']:
            value += residual[:,:,None,:].repeat(1,1,value.size(2),1)

        if self.config.fusion_config['query']:
            query_layer = self.query(query)
        else:
            query_layer = query

        if  self.config.fusion_config['key'] :
            key_layer = self.key(key)
        else:
            key_layer = key

        if self.config.fusion_config['value']:
            # key/value have dims => batch, toks, number-of-adapters, feats
            value_layer = self.value(value)
        else:
            value_layer = value


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.squeeze(torch.matmul(query_layer.unsqueeze(2), key_layer.transpose(-2,-1)), dim = 2)

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)

        self.T = max(self.T - self.reduction, 1.0)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # use the value layer or not TODO this is currently hardcoded
        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer),dim=2)
        # context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value), dim=2)

        if not self.config.fusion_config['residual_before']:
            context_layer += residual

        return context_layer


# class AdapterWeightingSentLvl(nn.Module):
#     def __init__(self, config, n_tasks):
#         super(AdapterWeightingSentLvl, self).__init__()
#         self.dense = nn.Linear(int(config.hidden_size), n_tasks)
#
#     def forward(self, query, key, value):
#         query_sent = torch.mean(query, dim=1)
#         scores = self.dense(query_sent)
#         probs = nn.Softmax(dim=-1)(scores)
#
#         weighted_value = probs.unsqueeze(1).unsqueeze(-1) * value
#         result = torch.sum(weighted_value, dim=2)
#
#         return result


# class AdapterWeightingSentLvlDynamic(nn.Module):
#     def __init__(self, config, n_tasks):
#         super(AdapterWeightingSentLvlDynamic, self).__init__()
#         self.dense = nn.Linear(int(config.hidden_size) // config.adapter_config['reduction_factor'], 1)
#
#         self.T = 50.0
#
#         self.reduction = self.T / 1000.0
#
#     def forward(self, query, key, value, attention_mask):
#
#         try:
#             attention_mask = (attention_mask == 0).float().to(key.device).squeeze()
#             length = torch.sum(attention_mask, dim=1)
#         except:
#             attention_mask = attention_mask.unsqueeze(1)
#             attention_mask = (attention_mask == 0).float().to(key.device)
#
#             length = torch.sum(attention_mask, dim=1)
#
#         attention_mask = attention_mask[:,:,None,None].repeat((1,1,key.size()[-2], key.size()[-1]))
#
#         key = key * attention_mask
#
#         key_sent = torch.sum(key, dim=1) / length[:,None,None].repeat(1,key.size()[-2], key.size()[-1])
#
#         # key_sent = torch.mean(key, dim=1)
#         scores = self.dense(key_sent)
#         scores_t = scores.transpose(-2, -1)
#         probs = nn.Softmax(dim=-1)(scores_t / self.T)
#         # attention_scores = attention_scores + attention_mask
#         #weighted_value = probs.unsqueeze(1).unsqueeze(-1) * value
#         #result = torch.sum(weighted_value, dim=2)
#
#         # with open('probabilities.txt', 'a') as f:
#         #     for b in probs.data.numpy():
#         #         f.write('\t'.join([str(e) for e in b[0]]) + '\n')
#
#         self.T = max(self.T - self.reduction, 1.0)
#
#         result = torch.squeeze(torch.matmul(probs.unsqueeze(2), value), dim=2)
#         return result


class AdapterFusionSentLvlDynamic(nn.Module):
    def __init__(self, config, n_tasks):
        super(AdapterFusionSentLvlDynamic, self).__init__()
        self.config = config

        self.dense_size = int(config.hidden_size) // config.adapter_config['reduction_factor']

        if not self.config.fusion_config['query'] and \
                not self.config.fusion_config['key'] and \
                not self.config.fusion_config['value']:
            self.dense = nn.Linear(self.dense_size, 1)

        if self.config.fusion_config['query'] :
            self.query = nn.Linear(int(config.hidden_size) , self.dense_size)

        if self.config.fusion_config['key'] :
            self.key = nn.Linear(self.dense_size, self.dense_size)

        if self.config.fusion_config['value'] :
            self.value =  nn.Linear(int(config.hidden_size), int(config.hidden_size))

        if self.config.fusion_config['temperature']:
            self.T = 50.0
        else:
            self.T = 1.0

        self.reduction = self.T / 1000.0


    def forward(self, query, key, value, attention_mask):

        attention_mask = (attention_mask == 0).float().to(key.device).squeeze()

        length = torch.sum(attention_mask, dim=1)

        # attention_mask = attention_mask[:,:,None,None].repeat((1,1,key.size()[-2], key.size()[-1]))

        key = key * attention_mask[:,:,None,None].repeat((1,1,key.size()[-2], key.size()[-1]))
        key_sent = torch.sum(key, dim=1) / length[:,None,None].repeat(1,key.size()[-2], key.size()[-1])

        if  self.config.fusion_config['query'] and \
                not self.config.fusion_config['key'] and \
                not self.config.fusion_config['value']:
            query = query * attention_mask[:,:,None].repeat((1,1, query.size()[-1]))
            query_sent = torch.sum(query, dim=1) / length[:,None].repeat(1, query.size()[-1])
            query_enc = self.query(query_sent)
            scores_t = torch.matmul(key_sent, query_enc[:, :, None]).squeeze(-1)
            probs = nn.Softmax(dim=-1)(scores_t / self.T)

            # result = torch.squeeze(torch.matmul(probs, value), dim=2)
            result = torch.squeeze(torch.matmul(probs[:,None,None,:], value))
        #     {'MR': {'devacc': 77.53, 'acc': 76.7, 'ndev': 9596, 'ntest': 9596}}
        if self.config.fusion_config['query'] and \
                self.config.fusion_config['key'] and \
                not self.config.fusion_config['value']:
            query = query * attention_mask[:,:,None].repeat((1,1, query.size()[-1]))
            query_sent = torch.sum(query, dim=1) / length[:,None].repeat(1, query.size()[-1])
            query_enc = self.query(query_sent)
            key_enc = self.key(key_sent)
            scores_t = torch.matmul(key_enc, query_enc[:, :, None]).squeeze(-1)
            probs = nn.Softmax(dim=-1)(scores_t / self.T)

            # result = torch.squeeze(torch.matmul(probs, value), dim=2)
            result = torch.squeeze(torch.matmul(probs[:,None,None,:], value))

        if self.config.fusion_config['query'] and \
                 self.config.fusion_config['key'] and \
                 self.config.fusion_config['value']:
            query = query * attention_mask[:,:,None].repeat((1,1, query.size()[-1]))
            query_sent = torch.sum(query, dim=1) / length[:,None].repeat(1, query.size()[-1])
            query_enc = self.query(query_sent)
            key_enc = self.key(key_sent)
            value_enc = self.value(value)
            scores_t = torch.matmul(key_enc, query_enc[:, :, None]).squeeze(-1)
            probs = nn.Softmax(dim=-1)(scores_t / self.T)

            # result = torch.squeeze(torch.matmul(probs, value), dim=2)
            result = torch.squeeze(torch.matmul(probs[:,None,None,:], value_enc))

        if not self.config.fusion_config['query'] and \
                not self.config.fusion_config['key'] and \
                not self.config.fusion_config['value']:
            # key_sent = torch.mean(key, dim=1)
            scores = self.dense(key_sent)
            scores_t = scores.transpose(-2, -1)

            probs = nn.Softmax(dim=-1)(scores_t / self.T)
            result = torch.squeeze(torch.matmul(probs.unsqueeze(2), value), dim=2)
        # attention_scores = attention_scores + attention_mask
        #weighted_value = probs.unsqueeze(1).unsqueeze(-1) * value
        #result = torch.sum(weighted_value, dim=2)

        self.T = max(self.T - self.reduction, 1.0)

        return result



# class AdapterWeightingSentLvlDynamic(nn.Module):
#     def __init__(self, config, n_tasks):
#         super(AdapterWeightingSentLvlDynamic, self).__init__()
#         self.dense = nn.Linear(int(config.hidden_size) // 2, 1)
#
#     def forward(self, query, key, value):
#         key_sent = torch.mean(key, dim=1)
#         scores = self.dense(key_sent)
#         scores_t = scores.transpose(-2, -1)
#         probs = nn.Softmax(dim=-1)(scores_t)
#
#         #weighted_value = probs.unsqueeze(1).unsqueeze(-1) * value
#         #result = torch.sum(weighted_value, dim=2)
#
#         result = torch.squeeze(torch.matmul(probs.unsqueeze(2), value), dim=2)
#         return result


class SimpleAdapterWeightingStatic(nn.Module):
    def __init__(self, config, n_tasks):
        super(SimpleAdapterWeightingStatic, self).__init__()
        self.weights = nn.Parameter(torch.ones(n_tasks), requires_grad=True)

    def forward(self, query, key, value):
        probs = nn.Softmax()(self.weights)

        weighted_value = torch.reshape(probs, [1, 1, -1, 1]) * value
        result = torch.sum(weighted_value, dim=2)

        return result


if __name__ == '__main__':
    adapter = Adapter(50, add_layer_norm=True )

    batch = torch.rand(16,50)

    print(adapter(batch))

