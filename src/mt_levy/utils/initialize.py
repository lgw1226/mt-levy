from typing import Optional
from logging import Logger

import gymnasium as gym
import metaworld

import wandb
from wandb.sdk.wandb_run import Run
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from mt_levy.utils.wrappers import SparseReward
from mt_levy.exploration_strategies import BaseExpStrategy


def initialize_envs(cfg: DictConfig, logger: Logger) -> gym.vector.VectorEnv:
    logger.info("Initializing environments")
    benchmark: str = cfg.environment.benchmark
    seed: int = cfg.seed
    sparse: bool = cfg.environment.sparse
    horizon: int = cfg.environment.horizon
    envs = gym.make_vec(
        f"Meta-World/{benchmark}",
        max_episode_steps=horizon,
        vector_strategy="async",
        seed=seed,
        terminate_on_success=sparse,
    )
    if sparse:
        envs = SparseReward(envs)
    return envs


def initialize_agent(cfg: DictConfig, logger: Logger):
    logger.info("Initializing agent")
    return instantiate(cfg.agent.builder, _recursive_=False)


def initialize_exploration_strategy(
    cfg: DictConfig, logger: Logger, agent
) -> BaseExpStrategy:
    logger.info("Initializing exploration strategy")
    return instantiate(cfg.exploration_strategy.builder, agent=agent)


def initialize_buffer(cfg: DictConfig, logger: Logger):
    logger.info("Initializing replay buffer")
    return instantiate(cfg.buffer.builder)


def initialize_wandb(cfg: DictConfig, logger: Logger) -> Optional[Run]:
    if not cfg.logging.use_wandb:
        return None
    logger.info("Initializing wandb")
    return wandb.init(
        **cfg.wandb,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
    )
