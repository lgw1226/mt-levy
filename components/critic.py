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
        super(Critic, self).__init__()
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

    # type hinting
    def __call__(self, obs: Tensor, act: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(obs, act)

    def forward(self, obs: Tensor, act: Tensor) -> tuple[Tensor, Tensor]:
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

        self.task_idx_to_mask = torch.eye(self.num_heads)
        self.q1 = self._make_q()
        self.q2 = self._make_q()
        self.apply(weight_init)

    # type hinting
    def __call__(self, obs: Tensor, act: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(obs, act, idx)

    def forward(self, obs: Tensor, act: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)
            idx = idx.unsqueeze(0)
        idx = idx.to(torch.int)
        
        mask = self._get_mask(idx)
        x = torch.cat([obs, act], dim=-1)
        q1 = torch.sum(self.q1(x) * mask, dim=0)
        q2 = torch.sum(self.q2(x) * mask, dim=0)
        return q1.squeeze(0, -1), q2.squeeze(0, -1)
    
    def _get_mask(self, task_idx: Tensor) -> Tensor:
        task_idx_to_mask = self.task_idx_to_mask.to(task_idx.device)
        mask = task_idx_to_mask[task_idx]
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return mask.t().unsqueeze(2).to(task_idx.device)

    def _make_q(self) -> nn.Module:
        trunk_arch = get_arch(self.in_dim, self.hidden_dim, self.hidden_dim, self.num_trunk_layers)
        trunk = MLP(trunk_arch)
        heads = FeedForward(self.num_heads, self.hidden_dim, self.out_dim, self.num_head_layers, self.hidden_dim)
        if self.kwargs.get('bound', False):
            return nn.Sequential(trunk, nn.ReLU(), heads, nn.Sigmoid())
        else:
            return nn.Sequential(trunk, nn.ReLU(), heads)
