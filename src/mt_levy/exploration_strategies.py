from typing import Any, Optional
from pathlib import Path

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
        self,
        agent: MTMHSAC,
        seed: Optional[int] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(agent, seed=seed)
        self.num_tasks: int = kwargs["num_tasks"]
        self.horizon: int = kwargs["horizon"]
        self.alpha_bar: float = kwargs["alpha_offset"]
        self.rho_bar: float = kwargs["sr_threshold"]
        self.num_neighbors: int = kwargs["num_neighbors"]

        metadata_path = Path(__file__).resolve().parent / "mt10_metadata.npy"
        mt10_metadata = np.load(metadata_path, allow_pickle=True)
        self.sequence_matrix = mt10_metadata[:, : self.num_neighbors]
        self.cnt = np.zeros(self.num_tasks, dtype=np.float32)
        self.act = self.np_random.uniform(
            -1, 1, (self.num_tasks, self.agent.act_dim)
        ).astype(np.float32)

    def get_action(self, obs: NDArray, success_ratio: NDArray) -> NDArray:
        high_success_idx = np.nonzero(success_ratio > self.rho_bar)[0]
        candidates = set(high_success_idx)
        alpha = self.alpha_bar + 1 / self.rho_bar ** (success_ratio / self.rho_bar)

        sample_idx = np.arange(self.num_tasks)
        sample = np.ones(self.num_tasks, dtype=np.bool_)
        for i in range(self.num_tasks):
            if success_ratio[i] >= self.rho_bar:
                continue
            if self.cnt[i] <= 1:
                self.cnt[i] = np.clip(
                    self.np_random.pareto(alpha[i]), 0, int(self.horizon * 0.1)
                )
                if self.cnt[i] > 1:
                    candidates_list = list(
                        (candidates & set(self.sequence_matrix[i])) | {i}
                    )
                    sample_idx[i] = self.np_random.choice(candidates_list)
            else:
                sample[i] = False

        self.cnt -= 1
        self.act[sample] = self.agent.get_action(obs, sample_idx)[sample]
        return self.act
