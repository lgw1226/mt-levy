from gymnasium.vector import VectorEnv, VectorWrapper


class SparseReward(VectorWrapper):
    def __init__(self, envs: VectorEnv):
        super().__init__(envs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = terminated.astype(float)
        return obs, reward, terminated, truncated, info
