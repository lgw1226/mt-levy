from typing import Optional
from logging import Logger

import wandb
from wandb.sdk.wandb_run import Run
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from envs.env_funcs import parse_benchmark
from exp_strategy import BaseExpStrategy


def initialize_envs(cfg: DictConfig, logger: Logger):
    logger.info("Initializing environments")
    return parse_benchmark(
        cfg.environment.benchmark,
        cfg.seed,
        sparse=cfg.environment.sparse,
        horizon=cfg.environment.horizon,
    )

def initialize_agent(cfg: DictConfig, logger: Logger):
    logger.info("Initializing agent")
    return instantiate(cfg.agent.builder, _recursive_=False)

def initialize_exploration_strategy(cfg: DictConfig, logger: Logger, agent) -> BaseExpStrategy:
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
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg),
    )
