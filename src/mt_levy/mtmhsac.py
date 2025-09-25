from typing import Union, Optional
from copy import deepcopy
from math import log

from hydra.utils import instantiate
import torch
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from omegaconf import DictConfig

from mt_levy.components.actor import MultiHeadActor
from mt_levy.components.critic import MultiHeadCritic
from mt_levy.components.utils import soft_update_params


class MTMHSAC:

    def __init__(
        self,
        num_envs: int,
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
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.target_temp = -act_dim

        self.gamma = gamma
        self.tau = tau
        self.device = (
            torch.device("cpu")
            if gpu_index is None
            else torch.device(f"cuda:{gpu_index}")
        )

        self.actor: MultiHeadActor = instantiate(actor_cfg, obs_dim, act_dim).to(
            self.device
        )
        self.critic: MultiHeadCritic = instantiate(critic_cfg, obs_dim, act_dim).to(
            self.device
        )
        self.critic_target: MultiHeadCritic = (
            deepcopy(self.critic).requires_grad_(False).to(self.device)
        )
        self.log_temp = Parameter(
            torch.tensor(self.num_envs * [log(init_temp)]).to(self.device)
        )
        self._components: dict[str, Module | Parameter] = {
            "actor": self.actor,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "log_temp": self.log_temp,
        }

        self.actor_optim: Optimizer = instantiate(
            actor_optim_cfg, params=self.actor.parameters()
        )
        self.critic_optim: Optimizer = instantiate(
            critic_optim_cfg, params=self.critic.parameters()
        )
        self.temp_optim: Optimizer = instantiate(temp_optim_cfg, params=[self.log_temp])

    @torch.no_grad()
    def get_action(
        self,
        obs: NDArray,
        idx: Optional[NDArray] = None,
        sample: bool = True,
    ) -> NDArray:
        if idx is None:
            idx = np.arange(self.num_envs)
        act, logp = self.actor(self._tensor(obs), self._tensor(idx), sample=sample)
        return self._ndarray(act)

    def update(self, batch: tuple[NDArray, ...]) -> dict[str, float]:
        obs, act, rwd, nobs, done, idx = map(self._tensor, batch)
        _act, _logp = self.actor(obs, idx)

        temp_loss, temp = self._update_temperature(_logp, idx)
        critic_loss = self._update_critics(obs, act, rwd, nobs, done, idx, temp)
        actor_loss = self._update_actor(obs, _act, _logp, idx, temp)
        self._update_targets()

        return {
            "train/loss/actor": actor_loss.item(),
            "train/loss/critic": critic_loss.item(),
            "train/loss/temperature": temp_loss.item(),
            "train/temperature": temp.mean().item(),
        }

    def _update_temperature(self, _logp: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
        idx = idx.to(torch.int)
        temp = torch.exp(self.log_temp)[idx]
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
        idx: Tensor,
        temp: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            nact, nlogp = self.actor(nobs, idx)
            nq1, nq2 = self.critic_target(nobs, nact, idx)
            nq = torch.minimum(nq1, nq2) - temp * nlogp
            td_target = rwd + (1 - done) * self.gamma * nq
        q1, q2 = self.critic(obs, act, idx)
        critic_loss = 0.5 * (F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss

    def _update_actor(
        self, obs: Tensor, _act: Tensor, _logp: Tensor, idx: Tensor, temp: Tensor
    ) -> Tensor:
        _q1, _q2 = self.critic(obs, _act, idx)
        _q = torch.minimum(_q1, _q2) - temp * _logp
        actor_loss = -torch.mean(_q)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return actor_loss

    def _update_targets(self):
        soft_update_params(self.critic.q1, self.critic_target.q1, self.tau)
        soft_update_params(self.critic.q2, self.critic_target.q2, self.tau)

    def save_ckpt(self, path: str):
        ckpt_dict = {
            "actor": self.actor.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "log_temp": self.log_temp.data,
            "temp_optim": self.temp_optim.state_dict(),
        }
        torch.save(ckpt_dict, path)

    def load_ckpt(self, path: str):
        ckpt_dict = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt_dict["actor"])
        self.actor_optim.load_state_dict(ckpt_dict["actor_optim"])
        self.critic.load_state_dict(ckpt_dict["critic"])
        self.critic_target.load_state_dict(ckpt_dict["critic_target"])
        self.critic_optim.load_state_dict(ckpt_dict["critic_optim"])
        self.log_temp.data = ckpt_dict[
            "log_temp"
        ]  # nn.Parameter does not have load_state_dict method
        self.temp_optim.load_state_dict(ckpt_dict["temp_optim"])

    def _tensor(self, data: NDArray) -> Tensor:
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def _ndarray(self, data: Tensor) -> NDArray:
        return data.detach().cpu().numpy()
