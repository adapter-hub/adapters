
from math import exp

import torch
import torch.nn as nn


def subnet_fc(c_in, c_out):
    # TODO activation param -> Activation_Function_Class -> relu default
    return nn.Sequential(nn.Linear(c_in, c_in//2), nn.ReLU(),
                         nn.Linear(c_in//2,  c_out))

class NICECouplingBlock(nn.Module):
    '''Coupling Block following the NICE design.
    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.'''

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None):
        super().__init__()

        channels = dims_in[0][0]
        self.split_len1 = channels // 2 # TODO hyperparam reduction
        self.split_len2 = channels - channels // 2

        assert all([dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        if not subnet_constructor:
            subnet_constructor = subnet_fc

        self.F = subnet_constructor(self.split_len2 + condition_length, self.split_len1)
        self.G = subnet_constructor(self.split_len1 + condition_length, self.split_len2)

    def forward(self, x, c=[], rev=False):
        # x1, x2 = (x[0].narrow(1, 0, self.split_len1),
        #           x[0].narrow(1, self.split_len1, self.split_len2))
        x1,x2 = (x[:,:,:self.split_len1],
                 x[:,:,self.split_len1:])
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
        # return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims



class GLOWCouplingBlock(nn.Module):
    '''Coupling Block following the GLOW design. The only difference to the RealNVP coupling blocks,
    is the fact that it uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate
    subnetworks. This reduces computational cost and speeds up learning.
    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.
    clamp:              Soft clamping for the multiplicative component. The amplification or attenuation
                        of each input dimension can be at most ±exp(clamp).'''

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=5.):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            F"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        if not subnet_constructor:
            subnet_constructor = subnet_fc

        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2*2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1*2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False):
        # x1, x2 = (x[0].narrow(1, 0, self.split_len1),
        #           x[0].narrow(1, self.split_len1, self.split_len2))

        x1,x2 = (x[:,:,:self.split_len1],
                 x[:,:,self.split_len1:])

        if not rev:
            # r2 = self.s2(torch.cat([x2, *c], 1) if self.conditional else x2)
            # s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            s2, t2 = x1.clone(), x2.clone()
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, *c], 1) if self.conditional else y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = (  torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims+1)))
                             + torch.sum(self.log_e(s2), dim=tuple(range(1, self.ndims+1))))

        else: # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, *c], 1) if self.conditional else x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(torch.cat([y2, *c], 1) if self.conditional else y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = (- torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims+1)))
                             - torch.sum(self.log_e(s2), dim=tuple(range(1, self.ndims+1))))

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims



