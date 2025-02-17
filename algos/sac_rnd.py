from typing import List, Tuple, Dict
from copy import deepcopy
from math import log
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
from components import MLP, create_mlp_layers
from utils import RunningMeanStd


class SACRND:

    log_std_min = -20
    log_std_max = 2

    def __init__(
            self,
            observation_dimension: int,
            action_dimension: int,
            actor_hidden_layers: List[int] = [256, 256],
            actor_lr: float = 3e-4,
            critic_hidden_layers: List[int] = [256, 256],
            critic_weight_gain: float = 0.001,
            critic_lr: float = 3e-4,
            init_temp: float = 1e-4,
            temp_lr: float = 1e-4,
            rnd_dimension: int = 64,
            rnd_hidden_layers: List[int] = [256, 256],
            rnd_lr: float = 3e-4,
            int_rwd_coef: float = 0.1,
            ext_gamma: float = 0.99,
            int_gamma: float = 0.5,
            tau: float = 0.005,
            device: torch.device = torch.device('cpu'),
    ):
        self.int_rwd_coef = int_rwd_coef
        self.ext_gamma = ext_gamma
        self.int_gamma = int_gamma
        self.tau = tau
        self.device = device

        actor_arch = [observation_dimension] + actor_hidden_layers + [action_dimension * 2]
        self.actor = MLP(actor_arch).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        critic_arch = [observation_dimension + action_dimension] + critic_hidden_layers + [1]
        self.ext_critic = DoubleCritic(critic_arch, bound_output=True, weight_gain=critic_weight_gain).to(self.device)
        self.ext_critic_optim = Adam(self.ext_critic.parameters(), lr=critic_lr)
        self.int_critic = DoubleCritic(critic_arch, bound_output=False, weight_gain=critic_weight_gain).to(self.device)
        self.int_critic_optim = Adam(self.int_critic.parameters(), lr=critic_lr)
    
        self.target_temp = -action_dimension
        self.log_temp = torch.tensor([log(init_temp)], requires_grad=True, device=self.device)
        self.temp_optim = Adam([self.log_temp], lr=temp_lr)

        rnd_arch = [observation_dimension] + rnd_hidden_layers + [rnd_dimension]
        self.rnd_net = MLP(rnd_arch).requires_grad_(False).to(self.device)
        self.rnd_pred = MLP(rnd_arch).to(self.device)
        self.rnd_optim = Adam(self.rnd_pred.parameters(), lr=rnd_lr)

        self.obs_stats = RunningMeanStd((observation_dimension,))
        self.rwd_stats = RunningMeanStd((1,))

    def save_ckpt(self, path: str):
        ckpt_dict = {
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'int_critic': self.int_critic.state_dict(),
            'int_critic_optim': self.int_critic_optim.state_dict(),
            'ext_critic': self.ext_critic.state_dict(),
            'ext_critic_optim': self.ext_critic_optim.state_dict(),
            'log_temp': self.log_temp,
            'temp_optim': self.temp_optim.state_dict(),
            'rnd_net': self.rnd_net.state_dict(),
            'rnd_pred': self.rnd_pred.state_dict(),
            'rnd_optim': self.rnd_optim.state_dict(),
            'obs_stats': self.obs_stats.__dict__(),
            'rwd_stats': self.rwd_stats.__dict__(),
        }
        torch.save(ckpt_dict, path)

    def load_ckpt(self, path: str):
        ckpt_dict = torch.load(path, weights_only=True)
        self.actor.load_state_dict(ckpt_dict['actor'])
        self.actor_optim.load_state_dict(ckpt_dict['actor_optim'])
        self.int_critic.load_state_dict(ckpt_dict['int_critic'])
        self.int_critic_optim.load_state_dict(ckpt_dict['critic_optim'])
        self.ext_critic.load_state_dict(ckpt_dict['ext_critic'])
        self.ext_critic_optim.load_state_dict(ckpt_dict['critic_optim'])
        self.log_temp = ckpt_dict['log_temp']
        self.temp_optim.load_state_dict(ckpt_dict['temp_optim'])
        self.rnd_net.load_state_dict(ckpt_dict['rnd_net'])
        self.rnd_pred.load_state_dict(ckpt_dict['rnd_pred'])
        self.rnd_optim.load_state_dict(ckpt_dict['rnd_optim'])
        self.obs_stats.mean = ckpt_dict['obs_stats']['mean']
        self.obs_stats.var = ckpt_dict['obs_stats']['var']
        self.obs_stats.count = ckpt_dict['obs_stats']['count']
        self.rwd_stats.mean = ckpt_dict['rwd_stats']['mean']
        self.rwd_stats.var = ckpt_dict['rwd_stats']['var']
        self.rwd_stats.count = ckpt_dict['rwd_stats']['count']

    @torch.no_grad()
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action, _ = self._get_action(self._tensor(observation), deterministic=deterministic)
        return self._ndarray(action)
    
    @torch.no_grad()
    def get_intrinsic_reward(self, next_observation: np.ndarray) -> float:
        normalized_obs = (next_observation - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + 1e-6)
        normalized_obs = torch.clip(self._tensor(normalized_obs), -5, 5)
        label = self.rnd_net(normalized_obs)
        pred = self.rnd_pred(normalized_obs)
        int_rwd = self._ndarray(torch.norm(label - pred, dim=-1, keepdim=True) ** 2)
        return int_rwd.item()
    
    @torch.no_grad()
    def update_obs_stats(self, next_observation: np.ndarray):
        """Update the observation statistics using the next observation.
        
        :param np.ndarray next_observation: The next observation of shape (obs_dim,).
        """
        next_observation += np.random.normal(0, 1e-3, next_observation.shape)
        self.obs_stats.update(next_observation[None, :])

    @torch.no_grad()
    def update_rwd_stats(self, next_observation: np.ndarray):
        """Update the intrinsic reward statistics using the next observation.
        
        :param np.ndarray next_observation: The next observation of shape (obs_dim,).
        """
        nobs = np.clip((next_observation - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + 1e-6), -5, 5)
        nobs_t = self._tensor(nobs)
        label = self.rnd_net(nobs_t)
        pred = self.rnd_pred(nobs_t)
        int_rwd = self._ndarray(torch.norm(label - pred, dim=-1, keepdim=True) ** 2)[None, :]
        self.rwd_stats.update(int_rwd)
    
    def update(self, batch: Tuple[np.ndarray, ...]) -> dict[str, float]:
        """Update the actor, critics, and temperature parameter using the batch of transitions.

        :param tuple[np.ndarray, ...] batch: The batch of transitions containing (obs, act, rwd, nobs, done).
        :return: The actor loss, critic loss, temperature loss, and temperature.
        :rtype: tuple[float, ...]
        """
        # unpack batch
        normalized_nobs_np = (batch[3] - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + 1e-6)
        normalized_nobs = torch.clip(self._tensor(normalized_nobs_np), -5, 5)
        obs, act, ext_rwd, nobs, done = map(self._tensor, batch)
        ext_rwd = ext_rwd.unsqueeze(-1)
        done = done.unsqueeze(-1)

        # compute intrinsic reward and normalize
        int_label = self.rnd_net(normalized_nobs)
        int_pred = self.rnd_pred(normalized_nobs)
        int_rwd = torch.norm(int_label - int_pred, dim=-1, keepdim=True) ** 2
        int_rwd = torch.clip(int_rwd / torch.sqrt(self._tensor(self.rwd_stats.var) + 1e-6), 0, 0.1)

        # update temperature
        _act, _logp = self._get_action(obs)
        temp = torch.exp(self.log_temp)
        temp_loss = -torch.mean(temp * (_logp + self.target_temp).detach())
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        temp = temp.clone().detach()

        # update critics
        # compute target Q values
        with torch.no_grad():
            nact, nlogp = self._get_action(nobs)
            int_nq1, int_nq2 = self.int_critic(nobs, nact, use_targets=True)
            int_nq = torch.minimum(int_nq1, int_nq2)
            int_targ = int_rwd + self.int_gamma * int_nq
            ext_nq1, ext_nq2 = self.ext_critic(nobs, nact, use_targets=True)
            ext_nq = torch.minimum(ext_nq1, ext_nq2) - temp * nlogp
            ext_targ = ext_rwd + (1 - done) * self.ext_gamma * ext_nq
            
        int_q1, int_q2 = self.int_critic(obs, act)
        int_critic_loss = 0.5 * torch.mean((int_q1 - int_targ) ** 2 + (int_q2 - int_targ) ** 2)
        self.int_critic_optim.zero_grad()
        int_critic_loss.backward()
        self.int_critic_optim.step()

        ext_q1, ext_q2 = self.ext_critic(obs, act)
        ext_critic_loss = 0.5 * torch.mean((ext_q1 - ext_targ) ** 2 + (ext_q2 - ext_targ) ** 2)
        self.ext_critic_optim.zero_grad()
        ext_critic_loss.backward()
        self.ext_critic_optim.step()

        # update actor
        _int_q1, _int_q2 = self.int_critic(obs, _act)
        _int_q = torch.minimum(_int_q1, _int_q2)
        _ext_q1, _ext_q2 = self.ext_critic(obs, _act)
        _ext_q = torch.minimum(_ext_q1, _ext_q2) - temp * _logp
        _q = _ext_q + self.int_rwd_coef * _int_q

        actor_loss = -torch.mean(_q)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update target networks
        self._update_targets()

        # update observation statistics after the update
        self.obs_stats.update(batch[3])

        # update rnd predictor
        label = self.rnd_net(normalized_nobs)
        pred = self.rnd_pred(normalized_nobs)
        rnd_loss = 0.5 * torch.mean(torch.norm(label - pred, dim=-1, keepdim=True) ** 2)
        self.rnd_optim.zero_grad()
        rnd_loss.backward()
        self.rnd_optim.step()

        # log
        log = {
            'train/loss/actor': actor_loss.item(),
            'train/loss/ext-critic': ext_critic_loss.item(),
            'train/ext-action-value': torch.mean(_ext_q).item(),
            'train/ext-rwd': torch.mean(ext_rwd).item(),
            'train/loss/int-critic': int_critic_loss.item(),
            'train/int-action-value': torch.mean(_int_q).item(),
            'train/int-rwd': torch.mean(int_rwd).item(),
            'train/loss/rnd': rnd_loss.item(),
            'train/loss/temperature': temp_loss.item(),
            'train/temperature': temp.item(),
        }

        return log

    def _update_targets(self):
        self.ext_critic.update_targets(self.tau)
        self.int_critic.update_targets(self.tau)

    def _get_action(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
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
        return squashed_action, squashed_log_prob
    
    def _bound_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        log_std = torch.tanh(log_std)
        scale = (self.log_std_max - self.log_std_min) / 2
        shift = (self.log_std_max + self.log_std_min) / 2
        return scale * log_std + shift
    
    def _tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def _ndarray(self, data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()