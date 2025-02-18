from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from components import get_arch, MLP, FeedForward
from components.utils import weight_init


class Critic(nn.Module):

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_dim: int,
            num_layers: int,
            **kwargs: dict[str, Any],
    ):
        super(MultiHeadCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.in_dim: int = obs_dim + act_dim
        self.out_dim = 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kwargs = kwargs

        self.q1 = self._make_q()
        self.q2 = self._make_q()
        self.apply(weight_init)

    def forward(self, obs: Tensor, act: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)

        x = torch.cat([obs, act], dim=-1)
        q1: Tensor = self.q1(x)
        q2: Tensor = self.q2(x)
        return q1.squeeze(0, -1), q2.squeeze(0, -1)

    def _make_q(self) -> nn.Module:
        model = MLP(get_arch(self.in_dim, self.out_dim, self.hidden_dim, self.num_layers))
        if self.kwargs.get('bound', False):
            return nn.Sequential(model, nn.Sigmoid())
        else:
            return model


class MultiHeadCritic(nn.Module):

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_dim: int,
            num_trunk_layers: int,
            num_heads: int,
            num_head_layers: int,
            **kwargs: dict[str, Any],
    ):
        super(MultiHeadCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.in_dim: int = obs_dim + act_dim
        self.out_dim = 1
        self.hidden_dim = hidden_dim
        self.num_trunk_layers = num_trunk_layers
        self.num_heads = num_heads
        self.num_head_layers = num_head_layers
        self.kwargs = kwargs

        self.q1 = self._make_q()
        self.q2 = self._make_q()
        self.apply(weight_init)

    def forward(self, obs: Tensor, act: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)
            idx = idx.unsqueeze(0)
        idx = idx.to(torch.int)
        aranged = torch.arange(obs.size(0), device=obs.device)

        x = torch.cat([obs, act], dim=-1)
        q1: Tensor = self.q1(x)[idx, aranged]
        q2: Tensor = self.q2(x)[idx, aranged]
        return q1.squeeze(0, -1), q2.squeeze(0, -1)

    def _make_q(self) -> nn.Module:
        trunk_arch = get_arch(self.in_dim, self.hidden_dim, self.hidden_dim, self.num_trunk_layers)
        trunk = MLP(trunk_arch)
        heads = FeedForward(self.num_heads, self.hidden_dim, self.out_dim, self.num_head_layers, self.hidden_dim)
        if self.kwargs.get('bound', False):
            return nn.Sequential(trunk, nn.ReLU(), heads, nn.Sigmoid())
        else:
            return nn.Sequential(trunk, nn.ReLU(), heads)
