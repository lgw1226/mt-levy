import os
import logging
from logging import Logger
from time import time
from typing import Optional

import torch
from wandb.sdk.wandb_run import Run
import hydra
from omegaconf import DictConfig
from tqdm import trange

from algos import SAC
from buffers import ReplayBuffer
from wrappers import MTWrapper
from utils import (
    initialize_envs,
    initialize_agent,
    initialize_buffer,
    initialize_wandb,
    save_ckpt,
)


# Setup logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    logger: Logger,
    epoch: int,
    env: MTWrapper,
    sac: SAC,
    buffer: ReplayBuffer,
    run: Optional[Run] = None,
):
    start = time()

    obs, _ = env.reset()
    for step in trange(1, cfg.training.train_steps + 1, desc=f"Epoch {epoch}", unit="step"):
        train_log = {"step": (epoch - 1) * cfg.training.train_steps + step}  # total steps

        # Select action: exploration during initial steps
        if epoch == 1 and step <= cfg.training.init_steps:
            act = env.action_space.sample()
        else:
            act = sac.get_action(obs)
        nobs, rwd, ter, tru, info = env.step(act)
        done = ter | tru

        # Store transitions (nobs could have been reset)
        buffer.append(obs, act, rwd, info['next_observation'], ter)
        obs = nobs

        # Training step
        if buffer.index >= cfg.training.batch_size or buffer.full:
            batch = buffer.sample(cfg.training.batch_size)
            fit_log = sac.update(batch)
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
    env: MTWrapper,
    sac: SAC,
    run: Optional[Run] = None,
):
    logger.info(f"Evaluating agent, epoch {epoch}")
    num_episodes: int = cfg.evaluation.num_episodes
    
    episode_cnt = 0
    success_cnt = 0
    total_rwd = 0
    total_step_cnt = 0

    success = False
    obs, info = env.reset()
    while episode_cnt < num_episodes:
        act = sac.get_action(obs, sample=False)
        obs, rwd, ter, tru, info = env.step(act)
        success = success | info['success']
        done = ter | tru
        
        episode_cnt += done
        if done and success:
            success_cnt += 1
            success = False
        total_rwd += rwd
        total_step_cnt += 1

    success_rate = success_cnt / num_episodes
    mean_return = total_rwd / num_episodes
    ep_len = total_step_cnt / num_episodes
                   
    if run:
        run.log({
            "eval/mean_return": mean_return,
            "eval/success_rate": success_rate,
            "eval/ep_len": ep_len,
            "epoch": epoch,
        })
    logger.info(
        f"Return: {mean_return:.2f} | "
        f"Success Rate: {success_rate:.2f} | "
        f"Episode Length: {ep_len:.2f}"
    )


@hydra.main(version_base=None, config_path="configs", config_name='train_sac')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    env: MTWrapper = initialize_envs(cfg, logger)
    sac: SAC = initialize_agent(cfg, logger)
    buffer: ReplayBuffer = initialize_buffer(cfg, logger)
    run = initialize_wandb(cfg, logger)

    logger.info("Starting training")
    for epoch in range(1, cfg.training.num_epochs + 1):
        train(cfg, logger, epoch, env, sac, buffer, run=run)
        evaluate(cfg, logger, epoch, env, sac, run=run)
        if epoch % cfg.logging.ckpt_interval == 0:
            save_ckpt(cfg, logger, epoch, sac, run=run)

    logger.info("Training complete")
    env.close()
    if run: run.finish()


if __name__ == "__main__":
    main()
