import math
from typing import Union

import torch
from torch import nn

from .configuration import AdapterFusionConfig
from .context import ForwardContext


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "gelu_orig":
            self.f = nn.functional.gelu
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)


# Single Adapter


class Adapter(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(
        self,
        adapter_name,
        input_size,
        down_sample=None,
        non_linearity="relu",
        init_bert_weights=True,
        add_layer_norm_before=True,
        add_layer_norm_after=False,
        residual_before_ln=True,
        **kwargs,
    ):
        super().__init__()
        self.name = adapter_name
        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        # ensure that the down sample size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        if kwargs.pop("phm_layer", False):
            # Linear down projection of the input
            seq_list.append(PHMLayer(self.input_size, self.down_sample, "down", **kwargs))
        else:
            seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        if kwargs.pop("phm_layer", False):
            # Linear down projection of the input
            self.adapter_up = PHMLayer(self.down_sample, self.input_size, "up" ** kwargs)
        else:
            self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if init_bert_weights:
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)

    def forward(self, x, residual_input):  # , residual_input=None):
        if self.name in ForwardContext.get_context().shared_parameters:
            parameters = ForwardContext.get_context().shared_parameters[self.name]
            # if len(parameters) > 0:
            # phm_parameters = nn.ParameterDict()
            # if "phm_rule" in parameters:
            #     phm_parameters["phm_rule"] = parameters["phm_rule"]
            # elif "phm_rule_left" in parameters:
            #     phm_parameters["phm_rule_left"] = parameters["phm_rule_left"]
            #     phm_parameters["phm_rule_right"] = parameters["phm_rule_right"]
            #
            # if "W_down" in parameters:
            #     phm_parameters["W"] = parameters["W_down"]
            # elif "W_down_left" in parameters:
            #     phm_parameters["W_left"] = parameters["W_down_left"]
            #     phm_parameters["W_right"] = parameters["W_down_right"]

            ForwardContext.get_context().phm_parameters = parameters
            down = self.adapter_down(x)

            # phm_parameters["W"] = None
            # phm_parameters["W_left"] = None
            # phm_parameters["W_right"] = None
            # if "W_up" in parameters:
            #     phm_parameters["W"] = parameters["W_up"]
            # elif "W_up_left" in parameters:
            #     phm_parameters["W_left"] = parameters["W_up_left"]
            #     phm_parameters["W_right"] = parameters["W_up_right"]
            # ForwardContext.get_context().phm_parameters = phm_parameters
            up = self.adapter_up(down)
        else:
            down = self.adapter_down(x)
            up = self.adapter_up(down)

        output = up

        # apply residual connection before layer norm if configured in this way
        if self.residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.residual_before_ln:
            output = output + residual_input

        return output, down, up

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


# Adapter Fusion


class BertFusion(nn.Module):
    """
    Implementation of an AdapterFusion block.
    """

    def __init__(
        self,
        config: AdapterFusionConfig,
        dense_size,
        attention_probs_dropout_prob,
    ):
        super(BertFusion, self).__init__()
        # if config.hidden_size % config.num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.config = config

        self.dense_size = dense_size
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        if not self.config["query"] and not self.config["key"] and not self.config["value"]:
            self.dense = nn.Linear(self.dense_size, 1)

        if self.config["query"]:
            self.query = nn.Linear(self.dense_size, self.dense_size)
            self.query.apply(Adapter.init_bert_weights)

        if self.config["key"]:
            self.key = nn.Linear(self.dense_size, self.dense_size)
            self.key.apply(Adapter.init_bert_weights)

        if self.config["value"]:
            self.value = nn.Linear(self.dense_size, self.dense_size, bias=False)
            self.value.apply(Adapter.init_bert_weights)
            if self.config["value_initialized"]:
                self.value.weight.data = (torch.zeros(self.dense_size, self.dense_size) + 0.000001).fill_diagonal_(1.0)

        if self.config["temperature"]:
            self.T = 50.0
        else:
            self.T = 1.0
        self.reduction = self.T / 1000.0

    def forward(self, query, key, value, residual):

        if self.config["residual_before"]:
            value += residual[:, :, None, :].repeat(1, 1, value.size(2), 1)

        if self.config["query"]:
            query_layer = self.query(query)
        else:
            query_layer = query

        if self.config["key"]:
            key_layer = self.key(key)
        else:
            key_layer = key

        if self.config["value"] and self.config["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            value_layer = self.value(value)
        else:
            value_layer = value

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.squeeze(torch.matmul(query_layer.unsqueeze(2), key_layer.transpose(-2, -1)), dim=2)

        attention_scores = self.dropout(attention_scores)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)
        self.T = max(self.T - self.reduction, 1.0)

        if not self.training:
            self.recent_attention = attention_probs.detach().cpu().numpy()

        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2)

        if self.config["value"] and not self.config["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            context_layer = self.value(context_layer)
        else:
            context_layer = context_layer

        if not self.config["residual_before"]:
            context_layer += residual

        return context_layer


# Invertible Adapters


def get_subnet_constructor(non_linearity, reduction_factor):
    def subnet(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, dims_in // reduction_factor),
            Activation_Function_Class(non_linearity),
            nn.Linear(dims_in // reduction_factor, dims_out),
        )

    return subnet


class NICECouplingBlock(nn.Module):
    """Coupling Block following the NICE design."""

    def __init__(self, dims_in, dims_c=[], non_linearity="relu", reduction_factor=2):
        super().__init__()

        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        assert all(
            [dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]
        ), "Dimensions of input and one or more conditions don't agree."
        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        subnet_constructor = get_subnet_constructor(non_linearity, reduction_factor)
        self.F = subnet_constructor(self.split_len2 + condition_length, self.split_len1)
        self.G = subnet_constructor(self.split_len1 + condition_length, self.split_len2)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[:, :, : self.split_len1], x[:, :, self.split_len1 :])
        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1 = x1 + self.F(x2_c)
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2 = x2 + self.G(y1_c)
        else:
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2 = x2 - self.G(x1_c)
            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1 = x1 - self.F(y2_c)

        return torch.cat((y1, y2), -1)

    def jacobian(self, x, rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class GLOWCouplingBlock(nn.Module):
    """
    Coupling Block following the GLOW design. The only difference to the RealNVP coupling blocks, is the fact that it
    uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate subnetworks. This reduces
    computational cost and speeds up learning. clamp: Soft clamping for the multiplicative component. The amplification
    or attenuation of each input dimension can be at most ±exp(clamp).
    """

    def __init__(self, dims_in, dims_c=[], non_linearity="relu", reduction_factor=2, clamp=5.0):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = math.exp(clamp)
        self.min_s = math.exp(-clamp)

        assert all(
            [tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]
        ), f"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."
        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        subnet_constructor = get_subnet_constructor(non_linearity, reduction_factor)
        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2 * 2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1 * 2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[:, :, : self.split_len1], x[:, :, self.split_len1 :])

        if not rev:
            s2, t2 = x1.clone(), x2.clone()
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, *c], 1) if self.conditional else y1)
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2 :]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims + 1))) + torch.sum(
                self.log_e(s2), dim=tuple(range(1, self.ndims + 1))
            )

        else:  # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, *c], 1) if self.conditional else x1)
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2 :]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(torch.cat([y2, *c], 1) if self.conditional else y2)
            s2, t2 = r2[:, : self.split_len1], r2[:, self.split_len1 :]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = -torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims + 1))) - torch.sum(
                self.log_e(s2), dim=tuple(range(1, self.ndims + 1))
            )

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims


def kronecker_product(a, b):
    """
    Copied from rabeehk/compacter seq2seq/hypercomplex/kronecker.py

    Kronecker product of matrices a and b with leading batch dimensions. Batch dimensions are broadcast. The number of
    them mush :type a: torch.Tensor :type b: torch.Tensor :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out


class PHMLayer(nn.Module):
    """
    This class is adapted from the compacter implementation at https://github.com/rabeehk/compacter
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        position: str,
        phm_dim: int,
        phm_rule: Union[None, torch.Tensor] = None,
        bias: bool = True,
        w_init: str = "normal",
        c_init: str = "normal",
        learn_phm: bool = True,
        shared_phm_rule=False,
        factorized_phm_W=False,
        shared_W_phm=False,
        factorized_phm_rule=False,
        phm_rank=1,
        phm_init_range=0.0001,
        kronecker_prod=False,
    ) -> None:
        super(PHMLayer, self).__init__()
        assert w_init in ["phm", "glorot-normal", "glorot-uniform", "normal"]
        assert c_init in ["normal", "uniform"]
        assert (
            in_features % phm_dim == 0
        ), f"Argument `in_features`={in_features} is not divisble be `phm_dim`{phm_dim}"
        assert (
            out_features % phm_dim == 0
        ), f"Argument `out_features`={out_features} is not divisble be `phm_dim`{phm_dim}"
        self.in_features = in_features
        self.out_features = out_features
        self.position = position
        self.learn_phm = learn_phm
        self.phm_dim = phm_dim
        self._in_feats_per_axis = in_features // phm_dim
        self._out_feats_per_axis = out_features // phm_dim
        self.phm_rank = phm_rank
        self.phm_rule = phm_rule
        self.phm_init_range = phm_init_range
        self.kronecker_prod = kronecker_prod
        self.shared_phm_rule = shared_phm_rule
        self.factorized_phm_rule = factorized_phm_rule
        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                self.phm_rule_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1), requires_grad=learn_phm)
                self.phm_rule_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim), requires_grad=learn_phm)
            else:
                self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim), requires_grad=learn_phm)
        self.bias_flag = bias
        self.w_init = w_init
        self.c_init = c_init
        self.shared_W_phm = shared_W_phm
        self.factorized_phm_W = factorized_phm_W
        if not self.shared_W_phm:
            if self.factorized_phm_W:
                self.W_left = nn.Parameter(
                    torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self.phm_rank)), requires_grad=True
                )
                self.W_right = nn.Parameter(
                    torch.Tensor(size=(phm_dim, self.phm_rank, self._out_feats_per_axis)), requires_grad=True
                )
            else:
                self.W = nn.Parameter(
                    torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self._out_feats_per_axis)), requires_grad=True
                )
        if self.bias_flag:
            self.b = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("b", None)
        self.reset_parameters()

    def init_W(self, W_left=None, W_right=None, W=None):
        if self.factorized_phm_W:
            W_left = W_left if W_left is not None else self.W_left
            W_right = W_right if W_right is not None else self.W_right
        else:
            W = W if W is not None else self.W
        if self.w_init == "glorot-normal":
            if self.factorized_phm_W:
                for i in range(self.phm_dim):
                    W_left.data[i] = nn.init.xavier_normal_(W_left.data[i])
                    W_right.data[i] = nn.init.xavier_normal_(W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    W.data[i] = nn.init.xavier_normal_(W.data[i])
        elif self.w_init == "glorot-uniform":
            if self.factorized_phm_W:
                for i in range(self.phm_dim):
                    W_left.data[i] = nn.init.xavier_uniform_(W_left.data[i])
                    W_right.data[i] = nn.init.xavier_uniform_(W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    W.data[i] = nn.init.xavier_uniform_(W.data[i])
        elif self.w_init == "normal":
            if self.factorized_phm_W:
                for i in range(self.phm_dim):
                    W_left.data[i].normal_(mean=0, std=self.phm_init_range)
                    W_right.data[i].normal_(mean=0, std=self.phm_init_range)
            else:
                for i in range(self.phm_dim):
                    W.data[i].normal_(mean=0, std=self.phm_init_range)
        else:
            raise ValueError

    def reset_parameters(self):
        if not self.shared_W_phm:
            self.init_W()

        if self.bias_flag:
            self.b.data = torch.zeros_like(self.b.data)

        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                if self.c_init == "uniform":
                    self.phm_rule_left.data.uniform_(-0.01, 0.01)
                    self.phm_rule_right.data.uniform_(-0.01, 0.01)
                elif self.c_init == "normal":
                    self.phm_rule_left.data.normal_(std=0.01)
                    self.phm_rule_right.data.normal_(std=0.01)
                else:
                    raise NotImplementedError
            else:
                if self.c_init == "uniform":
                    self.phm_rule.data.uniform_(-0.01, 0.01)
                elif self.c_init == "normal":
                    self.phm_rule.data.normal_(mean=0, std=0.01)
                else:
                    raise NotImplementedError

    def set_phm_rule(self, phm_rule=None, phm_rule_left=None, phm_rule_right=None):
        """
        If factorized_phm_rules is set, phm_rule is a tuple, showing the left and right phm rules, and if this is not
        set, this is showing the phm_rule.
        """
        if self.factorized_phm_rule:
            self.phm_rule_left = phm_rule_left
            self.phm_rule_right = phm_rule_right
        else:
            self.phm_rule = phm_rule

    def set_W(self, W=None, W_left=None, W_right=None):
        if self.factorized_phm_W:
            self.W_left = W_left
            self.W_right = W_right
        else:
            self.W = W

    def forward(self, x: torch.Tensor, phm_rule: Union[None, nn.ParameterList] = None) -> torch.Tensor:
        if self.shared_W_phm:
            parameters = ForwardContext.get_context().phm_parameters
            if self.factorized_phm_W:
                W = torch.bmm(parameters[f"W_{self.position}_left"], parameters[f"W_{self.position}_right"])
            else:
                W = parameters[f"W_{self.position}"]
        else:
            if self.factorized_phm_W:
                W = torch.bmm(self.W_left, self.W_right)
            else:
                W = self.W
        if self.shared_phm_rule:
            parameters = ForwardContext.get_context().phm_parameters
            if self.factorized_phm_rule:
                phm_rule = torch.bmm(parameters["phm_rule_left"], parameters["phm_rule_right"])
            else:
                phm_rule = parameters["phm_rule"]
        else:
            if self.factorized_phm_rule:
                phm_rule = torch.bmm(self.phm_rule_left, self.phm_rule_right)
            else:
                phm_rule = self.phm_rule

        if self.kronecker_prod:
            H = self.kronecker_product(phm_rule, W).sum(0)
        else:
            H = kronecker_product(phm_rule, W).sum(0)

        y = torch.matmul(input=x, other=H)
        if self.b is not None:
            y += self.b
        return y

    def init_shared_parameters(self):
        parameters = nn.ParameterDict()
        if self.shared_W_phm:
            if self.factorized_phm_W:
                W_down_left = torch.Tensor(size=(self.phm_dim, self._in_feats_per_axis, self.phm_rank))
                W_down_right = torch.Tensor(size=(self.phm_dim, self.phm_rank, self._out_feats_per_axis))
                W_up_left = torch.Tensor(size=(self.phm_dim, self._out_feats_per_axis, self.phm_rank))
                W_up_right = torch.Tensor(size=(self.phm_dim, self.phm_rank, self._in_feats_per_axis))
                self.init_W(W_left=W_down_left, W_right=W_down_right)
                self.init_W(W_left=W_up_left, W_right=W_up_right)
                parameters["W_down_left"] = nn.Parameter(W_down_left, requires_grad=True)
                parameters["W_down_right"] = nn.Parameter(W_down_right, requires_grad=True)
                parameters["W_up_left"] = nn.Parameter(W_up_left, requires_grad=True)
                parameters["W_up_right"] = nn.Parameter(W_up_right, requires_grad=True)
            else:
                W_down = torch.Tensor(size=(self.phm_dim, self._in_feats_per_axis, self._out_feats_per_axis))
                W_up = torch.Tensor(size=(self.phm_dim, self._out_feats_per_axis, self._in_feats_per_axis))
                self.init_W(W=W_down)
                self.init_W(W=W_up)
                parameters["W_down"] = nn.Parameter(W_down, requires_grad=True)
                parameters["W_up"] = nn.Parameter(W_up, requires_grad=True)
        if self.shared_phm_rule:
            if self.factorized_phm_rule:
                phm_rule_left = nn.Parameter(
                    torch.FloatTensor(self.phm_dim, self.phm_dim, 1).to(self.device), requires_grad=self.learn_phm
                )
                phm_rule_right = nn.Parameter(
                    torch.FloatTensor(self.phm_dim, 1, self.phm_dim).to(self.device), requires_grad=self.learn_phm
                )
                if self.c_init == "normal":
                    phm_rule_left.data.normal_(mean=0, std=self.phm_init_range)
                    phm_rule_right.data.normal_(mean=0, std=self.phm_init_range)
                elif self.c_init == "uniform":
                    phm_rule_left.data.uniform_(-1, 1)
                    phm_rule_right.data.uniform_(-1, 1)
                else:
                    raise NotImplementedError
                parameters["phm_rule_left"] = phm_rule_left
                parameters["phm_rule_right"] = phm_rule_right
            else:
                phm_rule = nn.Parameter(
                    torch.FloatTensor(self.phm_dim, self.phm_dim, self.phm_dim), requires_grad=self.learn_phm
                )
                if self.c_init == "normal":
                    phm_rule.data.normal_(mean=0, std=self.phm_init_range)
                elif self.c_init == "uniform":
                    phm_rule.data.uniform_(-1, 1)
                else:
                    raise NotImplementedError
                parameters["phm_rule"] = phm_rule
        return parameters
