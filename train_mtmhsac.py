import os
import logging
from time import time
from typing import Optional

import torch
import numpy as np
from numpy.typing import NDArray
import wandb
from wandb.sdk.wandb_run import Run
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from algos import MTMHSAC
from buffers import MTReplayBuffer
from envs import parse_benchmark
from utils import evaluate
from envs import SubprocVecEnv


# Setup logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)


def initialize_envs(cfg: DictConfig) -> SubprocVecEnv:
    logger.info("Initializing environments")
    return parse_benchmark(
        cfg.environment.benchmark,
        cfg.seed,
        sparse=cfg.environment.sparse,
        horizon=cfg.environment.horizon,
    )

def initialize_agent(cfg: DictConfig) -> MTMHSAC:
    logger.info("Initializing agent")
    return instantiate(cfg.agent.builder, _recursive_=False)

def initialize_buffer(cfg: DictConfig) -> MTReplayBuffer:
    logger.info("Initializing replay buffer")
    return MTReplayBuffer(
        cfg.buffer.buffer_size,
        cfg.environment.obs_dim,
        cfg.environment.act_dim,
        seed=cfg.seed,
    )

def initialize_wandb(cfg: DictConfig):
    if not cfg.logging.use_wandb:
        return None

    logger.info("Initializing wandb")
    return wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg),
    )

def train(
    cfg: DictConfig,
    epoch: int,
    success_rate: NDArray,
    env: SubprocVecEnv,
    mtmhsac: MTMHSAC,
    buffer: MTReplayBuffer,
    run: Optional[Run] = None,
):
    start = time()

    obs, _ = env.reset()
    for step in trange(1, cfg.training.train_steps + 1, desc=f"Epoch {epoch}", unit="step"):
        train_log = {"step": (epoch - 1) * cfg.training.train_steps + step}  # total steps

        # Select action: exploration during initial steps
        if epoch == 1 and step <= cfg.training.init_steps:
            act = env.sample_action()
        else:
            act = mtmhsac.get_action_all(obs)
        nobs, rwd, ter, tru, info = env.step(act)
        done = ter | tru

        # Update training success ratio when episode is done
        success = np.array([info[i]["success"] for i in range(env.num_envs)])
        success_rate[done] \
            = success_rate[done] * (1 - cfg.training.sr_decay) \
            + success[done] * cfg.training.sr_decay
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

@hydra.main(version_base=None, config_path="configs", config_name='train_mtmhsac')
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    # Initialize environments, buffer, and agent
    env = initialize_envs(cfg)
    mtmhsac = initialize_agent(cfg)
    buffer = initialize_buffer(cfg)
    run = initialize_wandb(cfg)

    logger.info("Starting training")
    success_rate = np.zeros(env.num_envs)
    for epoch in range(1, cfg.training.num_epochs + 1):
        # Train
        train(cfg, epoch, success_rate, env, mtmhsac, buffer, run=run)

        # Evaluate
        evaluate(cfg, epoch, env, mtmhsac, run=run)

        # Save checkpoint
        if epoch % cfg.logging.ckpt_interval == 0:
            ckpt_dir = os.path.join(cfg.logging.ckpt_base_dir, f"{run.name}-{run.id}" if run else "offline")
            os.makedirs(ckpt_dir, exist_ok=True)
            mtmhsac.save_ckpt(os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pt"))
            logger.info(f"Checkpoint saved at epoch {epoch}")

    # Cleanup
    logger.info("Training complete")
    env.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()
