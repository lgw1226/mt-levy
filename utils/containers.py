import numpy as np


class Trajectory:

    def __init__(self, max_path_length: int, obs_dim: int, act_dim: int, goal_dim: int):
        self.max_path_length = max_path_length
        self.ptr = 0

        self.observations = np.empty((max_path_length + 1, obs_dim), dtype=np.float32)
        self.achieved_goals = np.empty((max_path_length + 1, goal_dim), dtype=np.float32)
        self.actions = np.empty((max_path_length, act_dim), dtype=np.float32)
        self.rewards = np.empty((max_path_length, 1), dtype=np.float32)
        self.goals = np.empty((max_path_length, goal_dim), dtype=np.float32)

    def reset(self, observation: np.ndarray, achieved_goal: np.ndarray):
        self.ptr = 0
        self.observations[self.ptr] = observation
        self.achieved_goals[self.ptr] = achieved_goal

    def step(
            self,
            action: np.ndarray,
            reward: float,
            goal: np.ndarray,
            next_observation: np.ndarray,
            next_achieved_goal: np.ndarray
    ):
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.goals[self.ptr] = goal

        self.ptr += 1
        self.observations[self.ptr] = next_observation
        self.achieved_goals[self.ptr] = next_achieved_goal

    def get(self) -> dict[str, np.ndarray]:
        assert self.ptr == self.max_path_length
        return {
            'observations': self.observations,
            'achieved_goals': self.achieved_goals,
            'actions': self.actions,
            'rewards': self.rewards,
            'goals': self.goals
        }

class TrajectoryV2:
    """Store trajectories of fixed lengths. No goal space.
    
    :param int max_path_length: The maximum length of a trajectory.
    :param int obs_dim: The dimension of the observation space.
    :param int act_dim: The dimension of the action space.
    """
    def __init__(self, max_trajectory_length: int, obs_dim: int, act_dim: int):
        self.max_trajectory_length = max_trajectory_length
        self.ptr = 0

        self.observations = np.empty((max_trajectory_length + 1, obs_dim), dtype=np.float32)
        self.actions = np.empty((max_trajectory_length, act_dim), dtype=np.float32)
        self.rewards = np.empty((max_trajectory_length, 1), dtype=np.float32)

    def reset(self, observation: np.ndarray):
        """Reset the trajectory with the initial observation."""
        self.ptr = 0
        self.observations[self.ptr] = observation

    def step(
            self,
            action: np.ndarray,
            reward: float,
            next_observation: np.ndarray,
    ):
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward

        # increase the pointer
        self.ptr += 1
        self.observations[self.ptr] = next_observation

    def get(self) -> dict[str, np.ndarray]:
        """Return the trajectory as a dictionary."""
        assert self.ptr == self.max_trajectory_length
        return {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
        }
