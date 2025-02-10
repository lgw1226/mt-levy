from typing import Any
import numpy as np


class RunningMeanStd:

    def __init__(self, shape: tuple[int, ...]):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count: float = 1e-4

    def update(self, x: np.ndarray):
        """Update the running mean and standard deviation using the batch of data.
        
        :param np.ndarray x: The batch of data of shape (batch_size, *shape).
        """
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def __dict__(self) -> dict[str, Any]:
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count
        }
