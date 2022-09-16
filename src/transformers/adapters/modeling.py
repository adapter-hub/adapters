import math

import torch
from torch import nn

from transformers.activations import get_activation

from .configuration import AdapterConfig, AdapterFusionConfig
from .context import ForwardContext


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):
        super().__init__()
        if hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        else:
            self.f = get_activation(hidden_act.lower())

    def forward(self, x):
        return self.f(x)


# Single Adapter


class Adapter(nn.Module):
    """
    Implementation of a sequential bottleneck adapter block.
    """

    def __init__(
        self,
        adapter_name,
        input_size,
        down_sample,
        config: AdapterConfig,
    ):
        super().__init__()
        self.name = adapter_name
        self.input_size = input_size
        self.add_layer_norm_before = config["ln_before"]
        self.add_layer_norm_after = config["ln_after"]
        self.adapter_residual_before_ln = config["adapter_residual_before_ln"]
        self.use_gating = config["use_gating"]

        # Params related to input & output of adapter
        self.residual_before_ln = config["residual_before_ln"]
        self.original_ln_before = config["original_ln_before"]
        self.original_ln_after = config["original_ln_after"]

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

        if config["phm_layer"]:
            # Linear down projection of the input
            seq_list.append(PHMLayer(adapter_name, self.input_size, self.down_sample, "down", config))
        else:
            seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(config["non_linearity"].lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        if config["phm_layer"]:
            # Linear down projection of the input
            self.adapter_up = PHMLayer(adapter_name, self.down_sample, self.input_size, "up", config)
        else:
            self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # Additional scaling factor (from He et al. (2021))
        if isinstance(config["scaling"], float):
            self.scaling = config["scaling"]
        elif config["scaling"] == "learned":
            self.scaling = nn.Parameter(torch.ones(1))
        else:
            raise ValueError("Unknown scaling type: {}".format(config["scaling"]))

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        if self.use_gating:
            self.gate = nn.Linear(self.input_size, 1)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if config["init_weights"] == "bert":
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)
            if self.use_gating:
                self.gate.apply(self.init_bert_weights)
        elif config["init_weights"] == "mam_adapter":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.adapter_down[0].weight, a=math.sqrt(5))
                nn.init.zeros_(self.adapter_up.weight)
                nn.init.zeros_(self.adapter_down[0].bias)
                nn.init.zeros_(self.adapter_up.bias)
                if self.use_gating:
                    self.gate.apply(self.init_bert_weights)
        else:
            raise ValueError("Unknown init_weights type: {}".format(config["init_weights"]))

    def pre_forward(
        self,
        hidden_states,
        input_tensor,
        layer_norm,
        fusion_config=None,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration.

        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None

        if self.residual_before_ln:
            residual = hidden_states

        if fusion_config is not None and fusion_config["query_before_ln"]:
            query = hidden_states

        if self.original_ln_before:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        if not self.residual_before_ln:
            residual = hidden_states

        if fusion_config is not None and not fusion_config["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def forward(self, x, residual_input, output_gating=False):
        down = self.adapter_down(x)

        up = self.adapter_up(down)
        up = up * self.scaling
        output = up

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # apply residual connection before layer norm if configured in this way
        if self.adapter_residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.adapter_residual_before_ln:
            output = output + residual_input

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up

    def post_forward(self, hidden_states, input_hidden_states, input_tensor, layer_norm):
        """
        Performs computations after the forward pass of the adapter block(s). This e.g. includes applying the residual
        connection and layer norm if configured in this way.

        Args:
            hidden_states: The hidden states outputted by the adapter block(s).
            input_hidden_states: Residual connection before the adapter block(s).
            input_tensor: Residual connection before the Transformer FFN/ attention layer.
            layer_norm: Transformer LayerNorm.

        Returns:
            The modified hidden states.
        """
        if self.original_ln_after:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        return hidden_states

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


class ParallelAdapter(Adapter):
    """
    Implementation of a parallel bottleneck adapter block.
    """

    def __init__(self, adapter_name, input_size, down_sample, config: AdapterConfig):
        super().__init__(adapter_name, input_size, down_sample, config)

    def pre_forward(
        self,
        hidden_states,
        input_tensor,
        layer_norm,
        fusion_config=None,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration.

        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        # In case of parallel adapter, return the input tensor as hidden states
        query = None
        if fusion_config is not None:
            query = input_tensor
        return input_tensor, query, input_tensor

    def forward(self, x, residual_input, output_gating=False):
        down = self.adapter_down(x)

        up = self.adapter_up(down)
        up = up * self.scaling

        output = up

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up

    def post_forward(self, hidden_states, input_hidden_states, input_tensor, layer_norm):
        """
        Performs computations after the forward pass of the adapter block(s). This e.g. includes applying the residual
        connection and layer norm if configured in this way.

        Args:
            hidden_states: The hidden states outputted by the adapter block(s).
            input_hidden_states: Residual connection before the adapter block(s).
            input_tensor: Residual connection before the Transformer FFN/ attention layer.
            layer_norm: Transformer LayerNorm.

        Returns:
            The modified hidden states.
        """
        hidden_states = hidden_states + input_hidden_states

        if self.original_ln_after:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        return hidden_states


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

    def forward(self, query, key, value, residual, output_attentions: bool = False):

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

        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2)

        if self.config["value"] and not self.config["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            context_layer = self.value(context_layer)
        else:
            context_layer = context_layer

        if not self.config["residual_before"]:
            context_layer += residual

        if output_attentions:
            attention_probs = attention_probs.detach().cpu().numpy()
            return context_layer, attention_probs
        else:
            return context_layer


# Invertible Adapters


def get_subnet_constructor(non_linearity, reduction_factor):
    def subnet(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, int(dims_in // reduction_factor)),
            Activation_Function_Class(non_linearity),
            nn.Linear(int(dims_in // reduction_factor), dims_out),
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
        adapter_name: str,
        in_features: int,
        out_features: int,
        position: str,
        config: dict,
    ) -> None:
        super(PHMLayer, self).__init__()
        assert config["hypercomplex_nonlinearity"] in ["phm", "glorot-normal", "glorot-uniform", "normal"]
        assert config["phm_c_init"] in ["normal", "uniform"]
        assert (
            in_features % config["phm_dim"] == 0
        ), f"Argument `in_features`={in_features} is not divisble be `phm_dim`{config['phm_dim']}"
        assert (
            out_features % config["phm_dim"] == 0
        ), f"Argument `out_features`={out_features} is not divisble be `phm_dim`{config['phm_dim']}"
        self.name = adapter_name
        self.in_features = in_features
        self.out_features = out_features
        self.position = position
        self.learn_phm = config["learn_phm"]
        self.phm_dim = config["phm_dim"]
        self._in_feats_per_axis = in_features // config["phm_dim"]
        self._out_feats_per_axis = out_features // config["phm_dim"]
        self.phm_rank = config["phm_rank"]
        self.phm_init_range = config["phm_init_range"]
        self.shared_phm_rule = config["shared_phm_rule"]
        self.factorized_phm_rule = config["factorized_phm_rule"]
        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                self.phm_rule_left = nn.Parameter(
                    torch.FloatTensor(self.phm_dim, self.phm_dim, 1), requires_grad=self.learn_phm
                )
                self.phm_rule_right = nn.Parameter(
                    torch.FloatTensor(self.phm_dim, 1, self.phm_dim), requires_grad=self.learn_phm
                )
            else:
                self.phm_rule = nn.Parameter(
                    torch.FloatTensor(self.phm_dim, self.phm_dim, self.phm_dim), requires_grad=self.learn_phm
                )
        self.bias_flag = config["phm_bias"]
        self.w_init = config["hypercomplex_nonlinearity"]
        self.c_init = config["phm_c_init"]
        self.shared_W_phm = config["shared_W_phm"]
        self.factorized_phm_W = config["factorized_phm_W"]
        if not self.shared_W_phm:
            if self.factorized_phm_W:
                self.W_left = nn.Parameter(
                    torch.Tensor(size=(self.phm_dim, self._in_feats_per_axis, self.phm_rank)), requires_grad=True
                )
                self.W_right = nn.Parameter(
                    torch.Tensor(size=(self.phm_dim, self.phm_rank, self._out_feats_per_axis)), requires_grad=True
                )
            else:
                self.W = nn.Parameter(
                    torch.Tensor(size=(self.phm_dim, self._in_feats_per_axis, self._out_feats_per_axis)),
                    requires_grad=True,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared_W_phm:
            parameters = ForwardContext.get_context().shared_parameters[self.name]
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
            parameters = ForwardContext.get_context().shared_parameters[self.name]
            if self.factorized_phm_rule:
                phm_rule = torch.bmm(parameters["phm_rule_left"], parameters["phm_rule_right"])
            else:
                phm_rule = parameters["phm_rule"]
        else:
            if self.factorized_phm_rule:
                phm_rule = torch.bmm(self.phm_rule_left, self.phm_rule_right)
            else:
                phm_rule = self.phm_rule

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
