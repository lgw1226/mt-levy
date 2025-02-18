from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from omegaconf import DictConfig

from components import MLP, FeedForward, get_arch
from components.utils import weight_init


class Actor(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_dim: int,
            num_layers: int,
            log_std_max: float,
            log_std_min: float,
            **kwargs: dict[str, Any],
    ):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.in_dim = obs_dim
        self.out_dim = 2 * act_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.kwargs = kwargs

        self.model = self._make_model()
        self.apply(weight_init)

    def forward(
        self,
        obs: Tensor,
        sample: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        out: Tensor = self.model(obs)
        mean, log_std = out.chunk(2, dim=-1)
        std = _get_std(log_std, self.log_std_min, self.log_std_max)
        dist = Normal(mean, std)

        if sample:
            act = dist.rsample()
        else:
            act = mean
        logp: Tensor = dist.log_prob(act)
        
        squashed_act, squashed_logp = _squash(act, logp)
        return squashed_act.squeeze(0), squashed_logp.squeeze(0, -1)
    
    def _make_model(self) -> nn.Module:
        return MLP(get_arch(self.in_dim, self.out_dim, self.hidden_dim, self.num_layers))


class MultiHeadActor(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_dim: int,
            num_trunk_layers: int,
            num_heads: int,
            num_head_layers: int,
            log_std_max: float,
            log_std_min: float,
            **kwargs: dict[str, Any],
    ):
        super(MultiHeadActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.in_dim = obs_dim
        self.out_dim = 2 * act_dim
        self.hidden_dim = hidden_dim
        self.num_trunk_layers = num_trunk_layers
        self.num_heads = num_heads
        self.num_head_layers = num_head_layers
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.kwargs = kwargs

        self.model = self._make_model()
        self.apply(weight_init)

    def forward(
        self,
        obs: Tensor,
        idx: Tensor,
        sample: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            idx = idx.unsqueeze(0)
        idx = idx.to(torch.int)
        aranged = torch.arange(obs.size(0), device=obs.device)

        out: Tensor = self.model(obs)[idx, aranged]
        mean, log_std = out.chunk(2, dim=-1)
        std = _get_std(log_std, self.log_std_min, self.log_std_max)
        dist = Normal(mean, std)

        if sample:
            act = dist.rsample()
        else:
            act = mean
        logp: Tensor = dist.log_prob(act)
        
        squashed_act, squashed_logp = _squash(act, logp)
        return squashed_act.squeeze(0), squashed_logp.squeeze(0, -1)
    
    def _make_model(self) -> nn.Module:
        trunk = MLP(get_arch(self.in_dim, self.hidden_dim, self.hidden_dim, self.num_trunk_layers))
        heads = FeedForward(self.num_heads, self.hidden_dim, self.out_dim, self.num_head_layers, self.hidden_dim)
        model = nn.Sequential(trunk, nn.ReLU(), heads)
        return model


def _get_std(log_std: Tensor, log_std_min: float, log_std_max: float):
    """Return bounded standard deviation."""
    log_std = torch.tanh(log_std)
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
    return log_std.exp()

def _squash(act: Tensor, logp: Tensor):
    squashed_act = torch.tanh(act)
    squashed_logp = (logp - torch.log(1 - squashed_act.pow(2) + 1e-6)).sum(-1, keepdim=True)
    return squashed_act, squashed_logp
