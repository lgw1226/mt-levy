from typing import Optional
from math import log
from copy import deepcopy

import numpy as np
import torch
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.distributions import Normal

from hydra.utils import instantiate
from omegaconf import DictConfig

from components import Actor, Critic


class SAC:

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            actor_cfg: DictConfig,
            critic_cfg: DictConfig,
            actor_optim_cfg: DictConfig,
            critic_optim_cfg: DictConfig,
            temp_optim_cfg: DictConfig,
            init_temp: float = 0.1,
            gamma: float = 0.99,
            tau: float = 0.005,
            gpu_index: Optional[int] = None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.target_temp = -act_dim

        self.gamma = gamma
        self.tau = tau
        self.device = torch.device('cpu') if gpu_index is None else torch.device(f'cuda:{gpu_index}')

        self.actor: Actor = instantiate(actor_cfg, obs_dim, act_dim).to(self.device)
        self.critic: Critic = instantiate(critic_cfg, obs_dim, act_dim).to(self.device)
        self.critic_target: Critic = deepcopy(self.critic).requires_grad_(False).to(self.device)
        self.log_temp = Parameter(torch.tensor([log(init_temp)]).to(self.device))
        self._components: dict[Module, Parameter] = {
            'actor': self.actor,
            'critic': self.critic,
            'critic_target': self.critic_target,
            'log_temp': self.log_temp,
        }

        self.actor_optim: Optimizer = instantiate(actor_optim_cfg, params=self.actor.parameters())
        self.critic_optim: Optimizer = instantiate(critic_optim_cfg, params=self.critic.parameters())
        self.temp_optim: Optimizer = instantiate(temp_optim_cfg, params=[self.log_temp])

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, sample: bool = True) -> np.ndarray:
        act, logp = self.actor(self._tensor(obs), sample=sample)
        return self._ndarray(act)
    
    @torch.no_grad()
    def get_action_logs(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # convert observation to tensor
        observation = self._tensor(observation)

        # get mean and log_std from the actor
        mean, log_std = torch.chunk(self.actor(observation), 2, dim=-1)
        log_std = self._bound_log_std(log_std)
        dist = Normal(mean, torch.exp(log_std))

        # sample action and log probability from the normal distribution
        action = dist.rsample() if not deterministic else mean
        log_prob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        # squash the action and correct the log probability
        squashed_action = torch.tanh(action)
        squashed_log_prob = log_prob - torch.sum(torch.log(1 - squashed_action ** 2 + 1e-6), dim=-1, keepdim=True)

        return self._ndarray(squashed_action), self._ndarray(log_std)
    
    def update(self, batch: tuple[np.ndarray, ...]):
        # unpack batch
        obs, act, rwd, nobs, done = map(self._tensor, batch)
        rwd = rwd.unsqueeze(-1)
        done = done.unsqueeze(-1)

        # update temperature
        _act, _logp = self._get_action(obs)
        temp = torch.exp(self.log_temp)
        temp_loss = torch.mean(-torch.exp(self.log_temp) * (_logp + self.target_temp).detach())
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        temp = temp.detach()

        # update critics
        with torch.no_grad():
            nact, nlogp = self._get_action(nobs)
            nq1 = self.target1(torch.cat([nobs, nact], dim=-1))
            nq2 = self.target2(torch.cat([nobs, nact], dim=-1))
            nq = torch.minimum(nq1, nq2) - temp * nlogp
            td_target = rwd + (1 - done) * self.gamma * nq
        q1 = self.critic1(torch.cat([obs, act], dim=-1))
        q2 = self.critic2(torch.cat([obs, act], dim=-1))
        critic_loss = 0.5 * torch.mean((q1 - td_target) ** 2 + (q2 - td_target) ** 2)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        _q1 = self.critic1(torch.cat([obs, _act], dim=-1))
        _q2 = self.critic2(torch.cat([obs, _act], dim=-1))
        _q = torch.minimum(_q1, _q2)
        actor_loss = torch.mean(temp * _logp - _q)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update target networks
        self._update_targets()

        return actor_loss.item(), critic_loss.item(), temp_loss.item(), temp.item()

    def _update_targets(self):
        for target, critic in zip([self.target1, self.target2], [self.critic1, self.critic2]):
            csd = critic.state_dict()
            tsd = target.state_dict()
            for k in tsd.keys():
                tsd[k] = self.tau * csd[k] + (1 - self.tau) * tsd[k]
            target.load_state_dict(tsd)

    def _bound_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        log_std = torch.tanh(log_std)
        scale = (self.log_std_max - self.log_std_min) / 2
        shift = (self.log_std_max + self.log_std_min) / 2
        return scale * log_std + shift
    
    def _tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def _ndarray(self, data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()
    
    def save_ckpt(self, path: str):
        ckpt_dict = {
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'log_temp': self.log_temp.data,
            'temp_optim': self.temp_optim.state_dict(),
        }
        torch.save(ckpt_dict, path)

    def load_ckpt(self, path: str):
        ckpt_dict = torch.load(path)
        self.actor.load_state_dict(ckpt_dict['actor'])
        self.actor_optim.load_state_dict(ckpt_dict['actor_optim'])
        self.critic.load_state_dict(ckpt_dict['critic'])
        self.critic_target.load_state_dict(ckpt_dict['critic_target'])
        self.critic_optim.load_state_dict(ckpt_dict['critic_optim'])
        self.log_temp.data = ckpt_dict['log_temp']  # nn.Parameter does not have load_state_dict method
        self.temp_optim.load_state_dict(ckpt_dict['temp_optim'])