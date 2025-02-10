from typing import List, Optional

import gymnasium as gym
import numpy as np
from metaworld.envs.mujoco.sawyer_xyz import SawyerXYZEnv
from metaworld.types import Task


class MT1Wrapper(gym.Wrapper):
    def __init__(
            self,
            env: SawyerXYZEnv,
            tasks: List[Task],
            sparse_reward: bool = False,
            auto_reset: bool = False,
            max_path_length: int = 200,
            seed: int = None
    ):
        super(MT1Wrapper, self).__init__(env)
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
    
    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.sparse_reward:
            reward = float(info['success'])
            terminated = bool(info['success'])

        if (terminated or truncated) and self.auto_reset:
            new_obs, info = self.reset()
            info['final_observation'] = obs  # store the final observation
            obs = new_obs  # reset the environment
            
        return obs, reward, terminated, truncated, info
