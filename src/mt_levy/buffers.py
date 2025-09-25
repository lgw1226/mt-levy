from typing import Optional

import numpy as np
from numpy.typing import NDArray


class ReplayBuffer:
    def __init__(
        self, capacity: int, obs_dim: int, act_dim: int, seed: Optional[int] = None
    ):
        self.capacity = capacity
        self.idx = 0
        self.full = False
        self.np_random = np.random.default_rng(seed=seed)

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rwd = np.zeros((capacity,), dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.bool_)

    def append(
        self,
        obs: NDArray,
        act: NDArray,
        rwd: float,
        nobs: NDArray,
        done: bool,
    ):
        self.obs[self.idx] = obs
        self.act[self.idx] = act
        self.rwd[self.idx] = rwd
        self.nobs[self.idx] = nobs
        self.done[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int) -> tuple[NDArray, ...]:
        max_size = self.capacity if self.full else self.idx
        idxs = self.np_random.choice(max_size, batch_size, replace=False)
        return (
            self.obs[idxs],
            self.act[idxs],
            self.rwd[idxs],
            self.nobs[idxs],
            self.done[idxs],
        )


class MTReplayBuffer:
    def __init__(
        self, capacity: int, obs_dim: int, act_dim: int, seed: Optional[int] = None
    ):
        self.capacity = capacity
        self.index = 0
        self.full = False
        self.np_random = np.random.default_rng(seed=seed)

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rwd = np.zeros((capacity,), dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.bool_)
        self.tidx = np.zeros((capacity,), dtype=np.int32)

    def append(
        self,
        obs: NDArray,
        act: NDArray,
        rwd: NDArray,
        nobs: NDArray,
        done: NDArray,
        tidx: NDArray,
    ):
        """Efficiently store a batch of transitions using NumPy."""
        batch_size = obs.shape[0]
        idxs = np.arange(self.index, self.index + batch_size) % self.capacity

        self.obs[idxs] = obs
        self.act[idxs] = act
        self.rwd[idxs] = rwd
        self.nobs[idxs] = nobs
        self.done[idxs] = done
        self.tidx[idxs] = tidx

        self.index = (self.index + batch_size) % self.capacity
        self.full = self.full or self.index == 0

    def sample(self, batch_size: int) -> tuple[NDArray, ...]:
        """Efficiently sample a batch of experiences."""
        max_size = self.capacity if self.full else self.index
        idxs = self.np_random.choice(max_size, batch_size, replace=False)

        return (
            self.obs[idxs],
            self.act[idxs],
            self.rwd[idxs],
            self.nobs[idxs],
            self.done[idxs],
            self.tidx[idxs],
        )
