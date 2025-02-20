from typing import Optional
from math import log
from copy import deepcopy

from numpy import ndarray as NDArray
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from hydra.utils import instantiate
from omegaconf import DictConfig

from components.actor import Actor
from components.critic import Critic
from components.utils import soft_update_params


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
    def get_action(self, obs: NDArray, sample: bool = True) -> NDArray:
        act, logp = self.actor(self._tensor(obs), sample=sample)
        return self._ndarray(act)
    
    def update(self, batch: tuple[NDArray, NDArray, float, NDArray, bool]) -> dict[str, float]:
        obs, act, rwd, nobs, done = map(self._tensor, batch)
        _act, _logp = self.actor(obs)

        temp_loss, temp = self._update_temperature(_logp)
        critic_loss = self._update_critics(obs, act, rwd, nobs, done, temp)
        actor_loss = self._update_actor(obs, _act, _logp, temp)
        self._update_targets()

        return {
            'train/loss/actor': actor_loss.item(),
            'train/loss/critic': critic_loss.item(),
            'train/loss/temperature': temp_loss.item(),
            'train/temperature': temp.mean().item(),
        }
    
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
    
    def _update_temperature(self, _logp: Tensor) -> tuple[Tensor, Tensor]:
        temp = torch.exp(self.log_temp)
        temp_loss = torch.mean(-temp * (_logp.detach() + self.target_temp))
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        return temp_loss, temp.detach()
    
    def _update_critics(
            self,
            obs: Tensor,
            act: Tensor,
            rwd: Tensor,
            nobs: Tensor,
            done: Tensor,
            temp: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            nact, nlogp = self.actor(nobs)
            nq1, nq2 = self.critic_target(nobs, nact)
            nq = torch.minimum(nq1, nq2) - temp * nlogp
            td_target = rwd + (1 - done) * self.gamma * nq
        q1, q2 = self.critic(obs, act)
        critic_loss = 0.5 * (F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss
    
    def _update_actor(
            self,
            obs: Tensor,
            _act: Tensor,
            _logp:Tensor,
            temp: Tensor
    ) -> Tensor:
        _q1, _q2 = self.critic(obs, _act)
        _q = torch.minimum(_q1, _q2) - temp * _logp
        actor_loss = -torch.mean(_q)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return actor_loss

    def _update_targets(self):
        soft_update_params(self.critic.q1, self.critic_target.q1, self.tau)
        soft_update_params(self.critic.q2, self.critic_target.q2, self.tau)
    
    def _tensor(self, data: NDArray) -> Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def _ndarray(self, data: Tensor) -> NDArray:
        return data.detach().cpu().numpy()
