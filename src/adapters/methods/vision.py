# Module to support adapter training for vision related tasks

import torch.nn as nn


class StochasticDepth(nn.Module):
    """
    Applies Stochastic Depth (aka Drop Path) to residual networks.
    Constructed loosely upon the implementations in the `torchvision` library
    and the `timm` library.

    Randomly drops samples post-layer inside a batch via a `drop_prob` probability
    and scales them by `1-drop_prob` if the layer is kept if keep_prob_scaling is True.

    Paper: https://arxiv.org/pdf/1603.09382
    References: https://pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth

    """

    def __init__(self, drop_prob: float = 0.0, keep_prob_scaling: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob_scaling = keep_prob_scaling

    def forward(self, x):
        return stochastic_depth(
            x, self.drop_prob, self.keep_prob_scaling, self.training
        )


def stochastic_depth(
    x, drop_prob: float = 0.0, keep_prob_scaling: bool = True, training: bool = False
):
    """
    Applies stochastic_depth to a batch.

    Args:
        x: torch.Tensor of size (batch_size, ...)
            A residual block
        drop_prob: float between 0.0 <= drop_prob <= 1.0
            The probability of dropping the sample inside the batch
        keep_prob_scaling: bool, optional
            Boolean parameter to specify whether to scale samples by keep_prob if
            they are kept
        training: bool, optional
            Boolean parameter to specify whether or not the model is in training
            or inference mode. Stochastic Depth is not applied during inference
            similar to Dropout.
    """
    if drop_prob >= 1.0 or drop_prob < 0.0:
        raise ValueError("drop_prob must be between 0.0 and 1.0")

    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    # get the number of samples in the batch i.e input.shape[0]
    sample_shape = [x.shape[0]] + [1] * (x.ndim - 1)

    bernoulli_tensor = x.new_empty(
        sample_shape, dtype=x.dtype, device=x.device
    ).bernoulli_(keep_prob)
    if keep_prob_scaling:
        bernoulli_tensor.div_(keep_prob)

    return x * bernoulli_tensor
