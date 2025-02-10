from typing import List, Tuple, Dict
from copy import deepcopy
from math import log
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
from models import MLP, create_mlp_layers


class SparseSAC:
    log_std_min = -10
    log_std_max = 2

    def __init__(
            self,
            observation_dimension: int,
            action_dimension: int,
            actor_hidden_layers: List[int] = [256, 256],
            actor_lr: float = 0.0003,
            critic_hidden_layers: List[int] = [256, 256],
            critic_optimism: float = 0.2,
            critic_weight_gain: float = 0.1,
            critic_lr: float = 0.0003,
            init_temp: float = 0.0001,
            temp_lr: float = 0.0001,
            gamma: float = 0.99,
            tau: float = 0.005,
            device: torch.device = torch.device('cpu'),
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device

        actor_arch = [observation_dimension] + actor_hidden_layers + [action_dimension * 2]
        self.actor = MLP(actor_arch).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        def _init_weights(ml: nn.ModuleList) -> nn.ModuleList:
            for m in ml:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=critic_weight_gain)
                    nn.init.zeros_(m.bias)
            return ml
        
        assert 1e-3 <= critic_optimism and critic_optimism <= 1 - 1e-3
        class Sigmoid(nn.Module):
            def __init__(self): super().__init__()
            def forward(self, x): return torch.sigmoid((x - log(1 / critic_optimism - 1)))

        critic_arch = [observation_dimension + action_dimension] + critic_hidden_layers + [1]
        critic_layers = create_mlp_layers(critic_arch).append(Sigmoid())
        self.critic1 = nn.Sequential(*_init_weights(deepcopy(critic_layers))).to(self.device)
        self.target1 = deepcopy(self.critic1).requires_grad_(False).to(self.device)
        self.critic2 = nn.Sequential(*_init_weights(deepcopy(critic_layers))).to(self.device)
        self.target2 = deepcopy(self.critic2).requires_grad_(False).to(self.device)
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optim = Adam(critic_params, lr=critic_lr)

        self.target_temp = -action_dimension
        self.log_temp = torch.tensor([log(init_temp)], requires_grad=True, device=self.device)
        self.temp_optim = Adam([self.log_temp], lr=temp_lr)

    def save_ckpt(self, path: str):
        ckpt_dict = {
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic1': self.critic1.state_dict(),
            'target1': self.target1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target2': self.target2.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'log_temp': self.log_temp,
            'temp_optim': self.temp_optim.state_dict(),
        }
        torch.save(ckpt_dict, path)

    def load_ckpt(self, path: str):
        ckpt_dict = torch.load(path, weights_only=True)
        self.actor.load_state_dict(ckpt_dict['actor'])
        self.actor_optim.load_state_dict(ckpt_dict['actor_optim'])
        self.critic1.load_state_dict(ckpt_dict['critic1'])
        self.target1.load_state_dict(ckpt_dict['target1'])
        self.critic2.load_state_dict(ckpt_dict['critic2'])
        self.target2.load_state_dict(ckpt_dict['target2'])
        self.critic_optim.load_state_dict(ckpt_dict['critic_optim'])
        self.log_temp = ckpt_dict['log_temp']
        self.temp_optim.load_state_dict(ckpt_dict['temp_optim'])

    @torch.no_grad()
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action, _ = self._get_action(self._tensor(observation), deterministic=deterministic)
        return self._ndarray(action)
    
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
    
    def update(self, batch: Tuple[np.ndarray, ...]) -> Dict[str, float]:
        """Update the actor, critics, and temperature parameter using the batch of transitions.

        :param tuple[np.ndarray, ...] batch: The batch of transitions containing (obs, act, rwd, nobs, done).
        :return: The actor loss, critic loss, temperature loss, and temperature.
        :rtype: tuple[float, ...]
        """
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


        return {
            'train/loss/actor': actor_loss.item(),
            'train/loss/critic': critic_loss.item(),
            'train/loss/temperature': temp_loss.item(),
            'train/temperature': temp.item(),
        }

    def _update_targets(self):
        for target, critic in zip([self.target1, self.target2], [self.critic1, self.critic2]):
            csd = critic.state_dict()
            tsd = target.state_dict()
            for k in tsd.keys():
                tsd[k] = self.tau * csd[k] + (1 - self.tau) * tsd[k]
            target.load_state_dict(tsd)

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