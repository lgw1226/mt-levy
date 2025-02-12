from typing import List, Tuple, Dict
from copy import deepcopy
from math import log
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
from models import MLP, create_mlp_layers


class MTMHSAC:

    log_std_min = -10
    log_std_max = 2

    def __init__(
            self,
            num_heads: int,
            observation_dimension: int,
            action_dimension: int,
            actor_hidden_layers: List[int] = [400, 400, 400],
            actor_lr: float = 0.0003,
            critic_hidden_layers: List[int] = [400, 400, 400],
            bound_critic: bool = False,
            critic_optimism: float = 0.2,
            critic_weight_gain: float = 1,
            critic_lr: float = 0.0003,
            init_temp: float = 0.0001,
            temp_lr: float = 0.0001,
            gamma: float = 0.99,
            tau: float = 0.005,
            device: torch.device = torch.device('cpu'),
    ):
        self.num_heads = num_heads
        self.obs_dim = observation_dimension
        self.act_dim = action_dimension

        self.gamma = gamma
        self.tau = tau
        self.device = device

        self._init_actor(actor_hidden_layers, actor_lr)
        self._init_critics(critic_hidden_layers, bound_critic, critic_optimism, critic_weight_gain, critic_lr)
        self._init_temperature(init_temp, temp_lr)

    def _init_actor(self, actor_hidden_layers: List[int], actor_lr: float):
        actor_arch = [self.obs_dim] + actor_hidden_layers + [self.act_dim * 2 * self.num_heads]
        self.actor = MLP(actor_arch).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

    def _init_critics(self, critic_hidden_layers: List[int], bound_critic: bool, critic_optimism: float, critic_weight_gain: float, critic_lr: float):
        critic_arch = [self.obs_dim + self.act_dim] + critic_hidden_layers + [self.num_heads]
        def _init_weights(ml: nn.ModuleList) -> nn.ModuleList:
            for m in ml:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=critic_weight_gain)
                    nn.init.zeros_(m.bias)
            return ml
        if bound_critic:
            class Sigmoid(nn.Module):
                def __init__(self): super().__init__()
                def forward(self, x): return torch.sigmoid((x - log(1 / critic_optimism - 1)))
            critic_layers = create_mlp_layers(critic_arch).append(Sigmoid())
        else:
            critic_layers = create_mlp_layers(critic_arch)
        self.critic1 = nn.Sequential(*_init_weights(deepcopy(critic_layers))).to(self.device)
        self.target1 = deepcopy(self.critic1).requires_grad_(False).to(self.device)
        self.critic2 = nn.Sequential(*_init_weights(deepcopy(critic_layers))).to(self.device)
        self.target2 = deepcopy(self.critic2).requires_grad_(False).to(self.device)
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optim = Adam(critic_params, lr=critic_lr)

    def _init_temperature(self, init_temp: float, temp_lr: float):
        self.target_temp = -self.act_dim
        self.log_temp = torch.tensor(self.num_heads * [log(init_temp)], requires_grad=True, device=self.device)
        self.temp_optim = Adam([self.log_temp], lr=temp_lr)

    @torch.no_grad()
    def get_action(self, observation: np.ndarray, task_index: int, deterministic: bool = False) -> np.ndarray:
        action, _ = self._get_action(self._tensor(observation), self._tensor(task_index), deterministic=deterministic)
        return self._ndarray(action)
    
    def update(self, batch: Tuple[np.ndarray, ...]) -> Dict[str, float]:
        obs, act, rwd, nobs, done, idx = map(self._tensor, batch)
        rwd = rwd.unsqueeze(-1)
        done = done.unsqueeze(-1)
        idx = idx.to(torch.int)
        aranged = torch.arange(obs.size(0), device=self.device)

        temp_loss, temp = self._update_temperature(obs, idx)
        critic_loss = self._update_critics(obs, act, rwd, nobs, done, idx, aranged, temp)
        actor_loss = self._update_actor(obs, idx, aranged, temp)

        self._update_targets()

        return {
            'train/loss/actor': actor_loss.item(),
            'train/loss/critic': critic_loss.item(),
            'train/loss/temperature': temp_loss.item(),
            'train/temperature': temp.mean().item(),
        }

    def _update_temperature(self, obs: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _act, _logp = self._get_action(obs, idx)
        temp = torch.exp(self.log_temp)[idx]
        temp_loss = torch.mean(-torch.exp(self.log_temp) * (_logp + self.target_temp).detach())
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        return temp_loss, temp.detach()

    def _update_critics(self, obs: torch.Tensor, act: torch.Tensor, rwd: torch.Tensor, nobs: torch.Tensor, done: torch.Tensor, idx: torch.Tensor, aranged: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            nact, nlogp = self._get_action(nobs, idx)
            nq1 = self.target1(torch.cat([nobs, nact], dim=-1))[aranged, idx]
            nq2 = self.target2(torch.cat([nobs, nact], dim=-1))[aranged, idx]
            nq = torch.minimum(nq1, nq2) - temp * nlogp
            td_target = rwd + (1 - done) * self.gamma * nq
        q1 = self.critic1(torch.cat([obs, act], dim=-1))[aranged, idx]
        q2 = self.critic2(torch.cat([obs, act], dim=-1))[aranged, idx]
        critic_loss = 0.5 * torch.mean((q1 - td_target) ** 2 + (q2 - td_target) ** 2)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss

    def _update_actor(self, obs: torch.Tensor, idx: torch.Tensor, aranged: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        _act, _logp = self._get_action(obs, idx)
        _q1 = self.critic1(torch.cat([obs, _act], dim=-1))[aranged, idx]
        _q2 = self.critic2(torch.cat([obs, _act], dim=-1))[aranged, idx]
        _q = torch.minimum(_q1, _q2)
        actor_loss = torch.mean(temp * _logp - _q)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return actor_loss

    def _update_targets(self):
        for target, critic in zip([self.target1, self.target2], [self.critic1, self.critic2]):
            csd = critic.state_dict()
            tsd = target.state_dict()
            for k in tsd.keys():
                tsd[k] = self.tau * csd[k] + (1 - self.tau) * tsd[k]
            target.load_state_dict(tsd)

    def _get_action(self, observation: torch.Tensor, task_index: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        observation = observation.reshape(-1, self.obs_dim)
        task_index = task_index.to(torch.int)
        aranged = torch.arange(observation.size(0), device=self.device)

        mean, log_std = torch.chunk(self.actor(observation), 2, dim=-1)
        mean = mean.reshape(-1, self.num_heads, self.act_dim)[aranged, task_index]
        log_std = log_std.reshape(-1, self.num_heads, self.act_dim)[aranged, task_index]

        log_std = self._bound_log_std(log_std)
        dist = Normal(mean, torch.exp(log_std))

        action = dist.rsample() if not deterministic else mean
        log_prob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        squashed_action = torch.tanh(action)
        squashed_log_prob = log_prob - torch.sum(torch.log(1 - squashed_action ** 2 + 1e-6), dim=-1, keepdim=True)
        return squashed_action, squashed_log_prob

    def save_ckpt(self, path: str):
        ckpt_dict = {
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic1': self.critic1.state_dict(),
            'target1': self.target1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target2': self.target2.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'log_temp': self.log_temp.detach().cpu().numpy(),
            'temp_optim': self.temp_optim.state_dict(),
        }
        torch.save(ckpt_dict, path)

    def load_ckpt(self, path: str):
        ckpt_dict = torch.load(path)
        self.actor.load_state_dict(ckpt_dict['actor'])
        self.actor_optim.load_state_dict(ckpt_dict['actor_optim'])
        self.critic1.load_state_dict(ckpt_dict['critic1'])
        self.target1.load_state_dict(ckpt_dict['target1'])
        self.critic2.load_state_dict(ckpt_dict['critic2'])
        self.target2.load_state_dict(ckpt_dict['target2'])
        self.critic_optim.load_state_dict(ckpt_dict['critic_optim'])
        self.log_temp = torch.tensor(ckpt_dict['log_temp'], requires_grad=True, device=self.device)
        self.temp_optim.load_state_dict(ckpt_dict['temp_optim'])

    def _bound_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        log_std = torch.tanh(log_std)
        scale = (self.log_std_max - self.log_std_min) / 2
        shift = (self.log_std_max + self.log_std_min) / 2
        return scale * log_std + shift

    def _tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def _ndarray(self, data: torch.Tensor) -> np.ndarray:
        return data.detach().squeeze().cpu().numpy()
