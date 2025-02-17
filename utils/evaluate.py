import logging
from typing import Optional

import numpy as np
from wandb.sdk.wandb_run import Run
from omegaconf import DictConfig

from algos.mtmhsac import MTMHSAC
from envs.subproc_vec_env import SubprocVecEnv


logger = logging.getLogger(__name__)


def evaluate(
    cfg: DictConfig,
    epoch: int,
    envs: SubprocVecEnv,
    mtmhsac: MTMHSAC,
    run: Optional[Run] = None,
):
    """Evaluate the agent in the given environments for a number of episodes.

    Each environment runs exactly `num_episodes` episodes independently.
    """
    logger.info(f"Evaluating agent, epoch {epoch}")

    num_envs = envs.num_envs  # Number of parallel environments
    num_episodes: int = cfg.evaluation.num_episodes

    # Initialize tracking arrays
    ep_count = np.zeros(num_envs, dtype=int)  # Track completed episodes per env
    total_rwd = np.zeros(num_envs, dtype=float)  # Sum of rewards per env
    success_cnt = np.zeros(num_envs, dtype=int)  # Count of successful episodes per env
    total_step_cnt = np.zeros(num_envs, dtype=int)  # Track steps correctly

    # Reset environments
    obs, info = envs.reset()

    # Track which environments are still evaluating
    active_envs = np.ones(num_envs, dtype=np.bool_)  # True = still evaluating

    while np.any(active_envs):  # Only run until all environments finish num_episodes
        # Get actions from the agent
        act = mtmhsac.get_action_all(obs, sample=False)

        # Step through the environment
        obs, rwd, ter, tru, info = envs.step(act)
        done = ter | tru  # Compute done masks

        # Accumulate rewards **only for active environments**
        total_rwd[active_envs] += rwd[active_envs]
        total_step_cnt[active_envs] += 1

        # Track episode completions and successes
        ep_count += done  # Increment episode count where `done` is True
        done_and_active = done & active_envs
        for i in range(num_envs):
            if done_and_active[i]:  # Only process finished episodes
                if info[i]['success']:
                    success_cnt[i] += 1

        # Mark environments as **finished** if they have completed `num_episodes`
        active_envs = ep_count < num_episodes  # ✅ Mark finished environments as False

    # Compute metrics per environment
    mean_return = total_rwd / num_episodes
    success_rate = success_cnt / num_episodes
    ep_len = total_step_cnt / num_episodes

    # Log evaluation metrics
    if run:
        run.log({
            "eval/mean_return": mean_return.mean(),
            "eval/success_rate": success_rate.mean(),
            "eval/ep_len": ep_len.mean(),
            "epoch": epoch,
        })
    logger.info(
        f"Return: {mean_return.mean():.2f} | "
        f"Success Rate: {success_rate.mean():.2f} | "
        f"Episode Length: {ep_len.mean():.2f}"
    )
