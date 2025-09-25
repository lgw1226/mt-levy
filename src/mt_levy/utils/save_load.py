import os
from logging import Logger
from typing import Any, Optional

from wandb.sdk.wandb_run import Run
from omegaconf import DictConfig


def save_ckpt(cfg: DictConfig, logger: Logger, epoch: int, agent: Any, run: Optional[Run] = None):
    ckpt_dir = os.path.join(cfg.logging.ckpt_base_dir, f"{run.name}-{run.id}" if run else "offline")
    os.makedirs(ckpt_dir, exist_ok=True)
    agent.save_ckpt(os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pt"))
    logger.info(f"Checkpoint saved at epoch {epoch}")
