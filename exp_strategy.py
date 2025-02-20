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

    def get_action(self, obs: NDArray, **kwargs) -> NDArray:
        return self.agent.get_action(obs)


class QMP(BaseExpStrategy):

    def __init__(
            self,
            agent: MTMHSAC,
            seed: Optional[int] = None,
    ):
        super(QMP, self).__init__(agent, seed=seed)

    def get_action(self, obs: NDArray, **kwargs) -> NDArray:
        n_envs = len(obs)
        obs_repeat = obs.repeat(n_envs, axis=0)
        task_idx = np.tile(np.arange(n_envs), n_envs)
        act_repeat = self.agent.get_action(obs_repeat, task_idx)
        q_repeat = self._get_q(obs_repeat, act_repeat, task_idx).reshape(n_envs, n_envs)
        max_idx = np.argmax(q_repeat, axis=1)
        act = act_repeat.reshape(n_envs, n_envs, -1)[np.arange(n_envs), max_idx]
        return act
    
    def _get_q(self, obs: NDArray, act: NDArray, idx: NDArray) -> NDArray:
        obs, act, idx = map(self.agent._tensor, (obs, act, idx))
        q1, q2 = self.agent.critic(obs, act, idx)
        q1, q2 = map(self.agent._ndarray, (q1, q2))
        return np.minimum(q1, q2)
    

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
            if i not in candidate_idx: candidate_idx.append(i)
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
