from typing import List

import gymnasium as gym
import numpy as np
from metaworld.envs.mujoco.sawyer_xyz import SawyerXYZEnv
from metaworld.types import Task


class MTWrapper(gym.Wrapper):
    def __init__(
            self,
            env: SawyerXYZEnv,
            tasks: List[Task],
            sparse_reward: bool = False,
            auto_reset: bool = False,
            max_path_length: int = 200,
            seed: int = None
    ):
        super(MTWrapper, self).__init__(env)
        self.env = env
        self.env.max_path_length = max_path_length
        self.tasks = tasks
        self.sparse_reward = sparse_reward
        self.auto_reset = auto_reset
        self.np_random = np.random.default_rng(seed=seed)

    def reset(self, task_idx: int = None, **kwargs):
        if task_idx is not None:
            task = self.tasks[task_idx]
        else:
            task = self.tasks[self.np_random.choice(len(self.tasks))]
        self.env.set_task(task)
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, act: np.ndarray):
        obs, rwd, ter, tru, info = self.env.step(act)
        info['next_observation'] = obs
        if self.sparse_reward:
            rwd = float(info['success'])
            ter = bool(info['success'])

        if (ter or tru) and self.auto_reset:
            obs, _ = self.reset()
            
        return obs, rwd, ter, tru, info
