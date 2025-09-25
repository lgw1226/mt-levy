import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from mt_levy.components.mlp import Linear


def weight_init_linear(m: nn.Linear):
    assert isinstance(m.weight, Tensor)
    nn.init.xavier_uniform_(m.weight)
    assert isinstance(m.bias, Tensor)
    nn.init.zeros_(m.bias)


def weight_init_conv(m: nn.Conv2d):
    # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    assert isinstance(m.weight, Tensor)
    assert m.weight.size(2) == m.weight.size(3)
    m.weight.data.fill_(0.0)
    if hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)  # type: ignore[operator]
    mid = m.weight.size(2) // 2
    gain = nn.init.calculate_gain("relu")
    assert isinstance(m.weight, Tensor)
    nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def weight_init_moe_layer(m: Linear):
    assert isinstance(m.weight, Tensor)
    for i in range(m.weight.shape[0]):
        nn.init.xavier_uniform_(m.weight[i])
    assert isinstance(m.bias, Tensor)
    nn.init.zeros_(m.bias)


def weight_init(m: Module):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        weight_init_linear(m)
    elif isinstance(m, Linear):
        weight_init_moe_layer(m)


def soft_update_params(net: Module, target_net: Module, tau: float) -> None:
    """Perform soft udpate on the net using target net.

    Args:
        net ([ModelType]): model to update.
        target_net (ModelType): model to update with.
        tau (float): control the extent of update.
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.lerp_(param.data, tau)
