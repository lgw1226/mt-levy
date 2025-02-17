from typing import List
import torch
import torch.nn as nn
from torch import Tensor


__all__ = [
    'FeedForward',
    'MLP',
    'create_mlp_layers',
    'get_arch'
]


class Linear(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, bias: bool = True
    ):
        """torch.nn.Linear layer extended for use as a mixture of experts.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.rand(self.num_experts, self.in_features, self.out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_experts, 1, self.out_features))
            self.use_bias = True
        else:
            self.use_bias = False

    def forward(self, x: Tensor) -> Tensor:
        if self.use_bias:
            return x.matmul(self.weight) + self.bias
        else:
            return x.matmul(self.weight)

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class FeedForward(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        bias: bool = True,
    ):
        """A feedforward model of mixture of experts layers.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            num_layers (int): number of layers in the feedforward network.
            hidden_features (int): dimensionality of hidden layer in the
                feedforward network.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        layers: List[nn.Module] = []
        current_in_features = in_features
        for _ in range(num_layers - 1):
            linear = Linear(
                num_experts=num_experts,
                in_features=current_in_features,
                out_features=hidden_features,
                bias=bias,
            )
            layers.append(linear)
            layers.append(nn.ReLU())
            current_in_features = hidden_features
        linear = Linear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=out_features,
            bias=bias,
        )
        layers.append(linear)
        self._model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def __repr__(self) -> str:
        return str(self._model)


class MLP(nn.Module):
    def __init__(self, architecture: List[int]):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        for i in range(len(architecture) - 2):
            self.layers.append(nn.Linear(architecture[i], architecture[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(architecture[-2], architecture[-1]))

    def forward(self, x):
        return self.layers(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super(MLP, self).__call__(x)


def create_mlp_layers(architecture: List[int]) -> nn.ModuleList:
    layers = nn.ModuleList()
    for i in range(len(architecture) - 2):
        layers.append(nn.Linear(architecture[i], architecture[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(architecture[-2], architecture[-1]))
    return layers

def get_arch(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> List[int]:
    """Create a list of layer dimensions for a multi-layer perceptron."""
    return [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
