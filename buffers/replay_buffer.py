from typing import Tuple
from collections import deque
import numpy as np
from utils import Trajectory


class ReplayBuffer:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.buffer = deque(maxlen=max_len)

    def __len__(self):
        return len(self.buffer)
    
    def append(self, *args: np.ndarray):
        self.buffer.append(args)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = []
        for i in indices:
            batch.append(self.buffer[i])
        return tuple(map(np.stack, zip(*batch)))
