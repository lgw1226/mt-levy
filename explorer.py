from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from algos.mtmhsac import MTMHSAC


class Explorer(ABC):
    @abstractmethod
    def __init__(self, agent: MTMHSAC, seed: int):
        self.agent = agent
        self.n_tasks = agent.num_heads
        self.seed = seed
        self.np_random = np.random.default_rng(seed)

    @abstractmethod
    def get_action(self, obs: np.ndarray, task_idx: int):
        return self.agent.get_action(obs, task_idx)

class EGreedy(Explorer): pass

class EZGreedy(Explorer): pass

class MTEZGreedy(Explorer): pass

class Levy(Explorer): pass

class MTLevy(Explorer):
    def __init__(self, agent: MTMHSAC, seed: int, **kwargs: dict[str, Any]):
        super(MTLevy, self).__init__(agent, seed)
        self.is_exploring = np.zeros(self.n_tasks, dtype=bool)
        self.exp_idx = np.zeros(self.n_tasks, dtype=int)
        self.exp_cnt = np.zeros(self.n_tasks)
        self.exp_dur = np.zeros(self.n_tasks)

        self.max_exp_dur: int = kwargs['max_exp_dur'] if 'max_exp_dur' in kwargs else 100
        self.topn: int = kwargs['topn'] - 1 if 'topn' in kwargs else self.n_tasks  
        self.alpha_offset: float = kwargs['alpha_lb'] - 1

    def get_action(self, obs: np.ndarray, success_rate: np.ndarray) -> list[np.ndarray]:
        # topn tasks if task index is not included in candidate tasks
        # topn - 1 tasks if task index is included in candidate tasks
        candidate_idxs: list = np.argsort(success_rate)[-self.topn:].tolist()
        alpha = self.alpha_offset + self.agent.obs_dim ** success_rate  # success rate is bounded [0, 1]
        act = []
        for i in range(self.n_tasks):
            l = list(candidate_idxs)  # copy
            if i not in candidate_idxs: l.append(i)
            if not self.is_exploring[i]:
                step_size = self.np_random.pareto(alpha[i])
                if step_size < 2:
                    act.append(self.agent.get_action(obs, i))
                else:
                    self.is_exploring[i] = True
                    self.exp_dur[i] = np.clip(step_size, 2, self.max_exp_dur)
                    print(l)
                    self.exp_idx[i] = self.np_random.choice(l)
                    act.append(self.agent.get_action(obs, self.exp_idx[i]))
            else:
                act.append(self.agent.get_action(obs, self.exp_idx[i]))
                self.exp_cnt[i] += 1
                if self.exp_cnt[i] >= self.exp_dur[i]:
                    self.is_exploring[i] = False
                    self.exp_cnt[i] = 0
        return act


def set_explorer(agent: MTMHSAC, exp_type: str, seed: int, **kwargs):
    """Set the exploration strategy for the agent.

    :param MTMHSAC agent:
    :param str exp_type: Exploration strategy.
    :param kwargs: Additional arguments for the exploration strategy.
    """
    if exp_type == 'none':
        exp = Explorer(agent, seed)
    elif exp_type == 'mt-levy':
        exp = MTLevy(agent, seed, **kwargs)
    elif exp_type == 'mt-ez-greedy':
        exp = MTEZGreedy(agent, seed, **kwargs)
    else:
        raise ValueError(f'Invalid exploration type: {exp_type}')
    return exp


if __name__ == '__main__':

    import torch

    device = torch.device('cuda')
    agent = MTMHSAC(10, 39, 4)
    seed = 1
    exp_kwargs = {
        'max_exp_dur': 20,
        'topn': 3,
        'alpha_lb': 1.2
    }
    explorer = set_explorer(agent, 'mt-levy', seed, **exp_kwargs)
    success_rate = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * 0.1
    obs = np.random.rand(39)
    task_idx = 0
    act = explorer.get_action(obs, success_rate)
