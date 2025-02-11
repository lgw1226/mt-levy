import numpy as np
import gymnasium as gym
from algos import MTMHSAC


def evaluate(mtmhsac: MTMHSAC, eval_envs: list[gym.Env], num_episodes: int = 10) -> dict[str, np.ndarray]:
    """Evaluate the agent in the given environments for a number of episodes.
    The envrionments are reset automatically.

    :param MTMHSAC mtmhsac: The agent to evaluate.
    :param list[gym.Env] eval_envs: A list of gym environments.
    :param int num_episodes: The number of episodes to evaluate the agent.
    :return dict[str, np.ndarray]: A dictionary of evaluation metrics.
    """
    mean_return = []
    success_rate = []
    ep_len = []
    for i, env in enumerate(eval_envs):
        total_rwd = 0
        success_cnt = 0
        episode_cnt = 0
        total_step_cnt = 0
        obs, _ = env.reset()
        while episode_cnt < num_episodes:
            act = mtmhsac.get_action(obs, i, deterministic=True)
            obs, rwd, ter, tru, info = env.step(act)
            total_rwd += rwd
            total_step_cnt += 1
            if ter or tru:
                if info['success']:
                    success_cnt += 1
                episode_cnt += 1
        mean_return.append(total_rwd / num_episodes)
        success_rate.append(success_cnt / num_episodes)
        ep_len.append(total_step_cnt / num_episodes)

    return {
        'eval/mean-return': np.array(mean_return),
        'eval/success-rate': np.array(success_rate),
        'eval/episode-length': np.array(ep_len),
    }
