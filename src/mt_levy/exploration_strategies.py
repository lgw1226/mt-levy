from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from mt_levy.mtmhsac import MTMHSAC


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

    def __init__(
        self, agent: MTMHSAC, seed: Optional[int] = None, **kwargs: dict[str, Any]
    ):
        super(MTLevy, self).__init__(agent, seed=seed)
        self.num_tasks: int = kwargs["num_tasks"]
        self.topn: int = kwargs["top_n"]
        self.alpha_offset: float = kwargs["alpha_lower_bound"] - 1

        self.is_exp = np.zeros(self.num_tasks, dtype=np.bool_)
        self.idx = np.zeros(self.num_tasks, dtype=np.int32)
        self.cnt = np.zeros(self.num_tasks, dtype=np.float32)

    def get_action(self, obs: NDArray, success_rate: NDArray) -> NDArray:
        topn: set[int] = set(np.argsort(success_rate)[-self.topn :])
        high_success_idx = np.nonzero(success_rate > 0.8)
        alpha = self.alpha_offset + self.agent.obs_dim**success_rate

        # sample indices for exploration
        sample_idx = []
        for i in range(self.num_tasks):
            if not self.is_exp[i]:
                self.cnt[i] = self.np_random.pareto(alpha[i])
                if self.cnt[i] < 1:
                    sample_idx.append(i)
                else:
                    self.is_exp[i] = True
                    self.idx[i] = self.np_random.choice(list(topn | {i}))
                    sample_idx.append(self.idx[i])
            else:
                sample_idx.append(self.idx[i])
            self.cnt[i] -= 1
            if self.cnt[i] < 0:
                self.is_exp[i] = False

        # infer the actor
        sample_idx = np.array(sample_idx)
        return self.agent.get_action(obs, sample_idx)
