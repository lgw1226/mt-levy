from typing import Optional
from collections import deque
import numpy as np
from utils import Trajectory


class ReplayBuffer:
    def __init__(self, max_len: int, seed: int = None):
        self.max_len = max_len
        self.buffer = deque(maxlen=max_len)
        self.np_random = np.random.default_rng(seed=seed)

    def __len__(self):
        return len(self.buffer)
    
    def append(self, *args: np.ndarray):
        self.buffer.append(args)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        indices = self.np_random.choice(len(self.buffer), batch_size, replace=False)
        batch = []
        for i in indices:
            batch.append(self.buffer[i])
        return tuple(map(np.stack, zip(*batch)))


class MTReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: Optional[int] = None):
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

    def append(self,
               obs: np.ndarray,
               act: np.ndarray,
               rwd: np.ndarray,
               nobs: np.ndarray,
               done: np.ndarray,
               tidx: np.ndarray
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

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
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
