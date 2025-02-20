from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from envs.subproc_vec_env import SubprocVecEnv
from algos.mtmhsac import MTMHSAC


class BaseExpStrategy:

    def __init__(
            self,
            agent: MTMHSAC,
            seed: Optional[int] = None,
    ):
        self.agent = agent
        self.np_random = np.random.default_rng(seed=seed)

    def get_action(self, obs: NDArray) -> NDArray:
        return self.agent.get_action(obs)

# class EGreedy(Explorer): pass

# class EZGreedy(Explorer): pass

# class QMP(BaseExpStrategy):

#     def __init__(
#             self,
#             cfg: DictConfig,
#             env: SubprocVecEnv,
#             agent: MTMHSAC,
#             seed: Optional[int] = None,
#     ):
#         super(QMP, self).__init__(cfg, env, agent, seed=seed)

#     # def get_action(self, obs: tuple[np.ndarray]) -> np.ndarray:
#     #     actions = []
#     #     for i in range(self.n_tasks):
#     #         _act = self._get_action_qmp(obs[i])

#     #         for j in range(self.agent.num_heads):
#     #             _acts.append(self.agent.get_action(_obs, j))
#     #             _qs.append(self.agent.get_q(_obs, _acts[-1], j))
#     #         # select the action with the highest Q-value
#     #         actions.append(_acts[np.argmax(_qs)])
#     #     return actions
        
#     def _get_action_qmp(self, obs: np.ndarray) -> np.ndarray:
#         """Get action for QMP explorer.
        
#         Return actions for all tasks.
#         :param np.ndarray obs: obs of shape (obs_dim,).
#         """
#         obs = self.agent._tensor(obs)
#         mean, log_std = torch.chunk(self.agent.actor(obs), 2, dim=-1)
#         mean = mean.reshape(self.agent.num_heads, self.agent.act_dim)
#         log_std = log_std.reshape(self.agent.num_heads, self.agent.act_dim)
#         log_std = self.agent._bound_log_std(log_std)
#         dist = torch.distributions.Normal(mean, torch.exp(log_std))
#         action = dist.rsample()
#         squashed_action = torch.tanh(action)
#         return self.agent._ndarray(squashed_action)
    
#     def _get_q_values(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
#         """Get Q-values for QMP explorer.
        
#         Return Q-values for all tasks.
#         :param np.ndarray obs: obs of shape (obs_dim,).
#         :param np.ndarray action: Action of shape (n_tasks, act_dim).
#         """
#         obs = self.agent._tensor(obs).unsqueeze(0).repeat(self.n_tasks, 1)
#         action = self.agent._tensor(action)
#         q_values = []
#         for i in range(self.agent.num_heads):
#             q_values.append(self.agent.get_q(obs, action, i))
#         return self.agent._ndarray(q_values)


class MTLevy(BaseExpStrategy):

    def __init__(self, agent: MTMHSAC, seed: Optional[int] = None, **kwargs: dict[str, Any]):
        super(MTLevy, self).__init__(agent, seed=seed)
        self.num_tasks: int = kwargs['num_tasks']
        self.horizon: int = kwargs['horizon']
        self.max_exp_dur: float = kwargs.get('max_exploration_duration', self.horizon * 0.2)
        self.topn: int = kwargs.get('topn', 5)
        self.alpha_offset: float = kwargs.get('alpha_lb', 1) - 1

        self.is_exploring = np.zeros(self.num_tasks, dtype=np.bool_)
        self.exp_idx = np.zeros(self.num_tasks, dtype=np.int32)
        self.exp_cnt = np.zeros(self.num_tasks, dtype=np.float32)
        self.exp_dur = np.zeros(self.num_tasks, dtype=np.float32)

    def get_action(self, obs: NDArray, success_rate: NDArray) -> NDArray:
        candidate_idx = list(np.argsort(success_rate)[-self.topn:])
        alpha = self.alpha_offset + self.agent.obs_dim ** success_rate

        # sample indices for exploration
        idx = []
        for i in range(self.num_tasks):
            candidate_idx.append(i)
            if not self.is_exploring[i]:
                step_size = self.np_random.pareto(alpha[i])
                if step_size < 2:
                    idx.append(i)
                else:
                    self.is_exploring[i] = True
                    self.exp_dur[i] = np.clip(step_size, 2, self.max_exp_dur)
                    self.exp_idx[i] = self.np_random.choice(list(candidate_idx))
                    idx.append(self.exp_idx[i])
            else:
                idx.append(self.exp_idx[i])
                self.exp_cnt[i] += 1
                if self.exp_cnt[i] >= self.exp_dur[i]:
                    self.is_exploring[i] = False
                    self.exp_cnt[i] = 0
            candidate_idx.pop()

        # infer the actor
        idx = np.array(idx)
        return self.agent.get_action(obs, idx)
