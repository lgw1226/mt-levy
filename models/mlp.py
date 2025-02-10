from typing import List
import torch
import torch.nn as nn


__all__ = [
    'MLP',
    'create_mlp_layers',
]


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
