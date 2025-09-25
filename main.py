import os
import logging
from logging import Logger
from time import time
from typing import Optional

from gymnasium.vector import AsyncVectorEnv
import torch
import numpy as np
from numpy.typing import NDArray
import hydra
from omegaconf import DictConfig
from wandb.sdk.wandb_run import Run
from tqdm import trange

from mt_levy.mtmhsac import MTMHSAC
from mt_levy.buffers import MTReplayBuffer
from mt_levy.utils import (
    initialize_envs,
    initialize_agent,
    initialize_buffer,
    initialize_exploration_strategy,
    initialize_wandb,
    save_ckpt,
)
from exp_strategy import BaseExpStrategy


# Setup logging
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    logger: Logger,
    epoch: int,
    success_rate: NDArray,
    env: AsyncVectorEnv,
    mtmhsac: MTMHSAC,
    exp_strategy: BaseExpStrategy,
    buffer: MTReplayBuffer,
    run: Optional[Run] = None,
):
    start = time()

    obs, _ = env.reset()
    success = np.zeros(env.num_envs, dtype=np.bool_)
    for step in trange(
        1, cfg.training.train_steps + 1, desc=f"Epoch {epoch}", unit="step"
    ):
        train_log = {
            "step": (epoch - 1) * cfg.training.train_steps + step
        }  # total steps

        # Select action: exploration during initial steps
        if epoch == 1 and step <= cfg.training.init_steps:
            act = env.action_space.sample()
        else:
            act = exp_strategy.get_action(obs, success_rate=success_rate)
        nobs, rwd, ter, tru, info = env.step(act)
        done = ter | tru

        # Update training success ratio when episode is done
        success = np.logical_or(
            success, np.array([info[i]["success"] for i in range(env.num_envs)])
        )
        success_rate[done] = (
            success_rate[done] * (1 - cfg.training.sr_decay)
            + success[done] * cfg.training.sr_decay
        )
        success[done] = False
        train_log["train/success-rate"] = success_rate.mean()

        # Store transitions (nobs could have been reset)
        _nobs = np.array([info[i]["next_observation"] for i in range(env.num_envs)])
        buffer.append(obs, act, rwd, _nobs, ter, np.arange(env.num_envs))
        obs = nobs

        # Training step
        if buffer.index >= cfg.training.batch_size or buffer.full:
            batch = buffer.sample(cfg.training.batch_size)
            fit_log = mtmhsac.update(batch)
            train_log.update(fit_log)

        # Log training metrics
        if step % cfg.logging.log_interval == 0 and run is not None:
            run.log(train_log)

    end = time()
    logger.info(f"Epoch {epoch} complete | Elapsed Time: {end - start:.2f} (seconds)")


def evaluate(
    cfg: DictConfig,
    logger: Logger,
    epoch: int,
    env: AsyncVectorEnv,
    sac: MTMHSAC,
    run: Optional[Run] = None,
):
    logger.info(f"Evaluating agent, epoch {epoch}")

    num_envs = env.num_envs  # Number of parallel environments
    num_episodes: int = cfg.evaluation.num_episodes

    # Initialize tracking arrays
    ep_count = np.zeros(num_envs, dtype=int)  # Track completed episodes per env
    total_rwd = np.zeros(num_envs, dtype=float)  # Sum of rewards per env
    success_cnt = np.zeros(num_envs, dtype=int)  # Count of successful episodes per env
    total_step_cnt = np.zeros(num_envs, dtype=int)  # Track steps correctly

    # Reset environments
    success = np.zeros(num_envs, dtype=np.bool_)
    obs, info = env.reset()

    # Track which environments are still evaluating
    active_envs = np.ones(num_envs, dtype=np.bool_)  # True = still evaluating

    while np.any(active_envs):  # Only run until all environments finish num_episodes
        # Get actions from the agent
        act = sac.get_action(obs, sample=False)

        # Step through the environment
        obs, rwd, ter, tru, info = env.step(act)
        success = np.logical_or(
            success, np.array([info[i]["success"] for i in range(num_envs)])
        )
        done = ter | tru  # Compute done masks

        # Accumulate rewards **only for active environments**
        total_rwd[active_envs] += rwd[active_envs]
        total_step_cnt[active_envs] += 1

        # Track episode completions and successes
        ep_count += done  # Increment episode count where `done` is True
        done_and_active = done & active_envs
        for i in range(num_envs):
            if done_and_active[i] and success[i]:  # Only process finished episodes
                success_cnt[i] += 1
        success[done] = False

        # Mark environments as **finished** if they have completed `num_episodes`
        active_envs = ep_count < num_episodes  # ✅ Mark finished environments as False

    # Compute metrics per environment
    mean_return = total_rwd / num_episodes
    success_rate = success_cnt / num_episodes
    ep_len = total_step_cnt / num_episodes

    # Log evaluation metrics
    if run:
        run.log(
            {
                "eval/mean_return": mean_return.mean(),
                "eval/success_rate": success_rate.mean(),
                "eval/ep_len": ep_len.mean(),
                "epoch": epoch,
            }
        )
    logger.info(
        f"Return: {mean_return.mean():.2f} | "
        f"Success Rate: {success_rate.mean():.2f} | "
        f"Episode Length: {ep_len.mean():.2f}"
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    # Initialize environments, buffer, and agent
    env: AsyncVectorEnv = initialize_envs(cfg, logger)
    mtmhsac: MTMHSAC = initialize_agent(cfg, logger)
    buffer: MTReplayBuffer = initialize_buffer(cfg, logger)
    exp_strategy = initialize_exploration_strategy(cfg, logger, mtmhsac)
    run = initialize_wandb(cfg, logger)

    logger.info("Starting training")
    success_rate = np.zeros(env.num_envs)
    for epoch in range(1, cfg.training.num_epochs + 1):
        train(
            cfg,
            logger,
            epoch,
            success_rate,
            env,
            mtmhsac,
            exp_strategy,
            buffer,
            run=run,
        )
        evaluate(cfg, logger, epoch, env, mtmhsac, run=run)
        if epoch % cfg.logging.ckpt_interval == 0:
            save_ckpt(cfg, logger, epoch, mtmhsac, run=run)

    # Cleanup
    logger.info("Training complete")
    env.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()
