import os
import logging
from typing import Union
from datetime import datetime

import torch
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import trange

from algos import MTMHSAC
from buffers import ReplayBuffer
from envs import parse_benchmark
from utils import evaluate

os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='train_mtmhsac')
def main(cfg: DictConfig) -> None:

    # logger.info(f'configuration\n{OmegaConf.to_yaml(cfg)}')
    logger.info('unpack configuration')
    seed: int = cfg.seed
    torch.random.manual_seed(seed)
    gpu_index: int = cfg.gpu_index
    device = f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu'
    benchmark: Union[str, ListConfig] = cfg.benchmark
    sparse: bool = cfg.sparse
    horizon: int = cfg.horizon

    num_epochs: int = cfg.num_epochs  # evaluate after each epoch
    eval_episodes: int = cfg.eval_episodes  # evaluate for this many episodes
    init_steps: int = cfg.init_steps  # only for the initial epoch
    train_steps: int = cfg.train_steps  # per epoch
    log_interval: int = cfg.log_interval
    buffer_size: int = cfg.buffer_size
    batch_size: int = cfg.batch_size

    exp_type: str = cfg.exploration.type
    sr_decay: float = cfg.exploration.success_rate_decay

    logger.info('initialize environments, agent, and buffer')
    vec_env, eval_envs, obs_dim, act_dim = parse_benchmark(benchmark, seed, sparse=sparse, horizon=horizon)
    mtmhsac = MTMHSAC(vec_env.num_envs, obs_dim, act_dim, **cfg.mtmhsac, device=device)
    buffer = ReplayBuffer(buffer_size, seed=seed)
    if exp_type == 'none':
        pass
    elif exp_type == 'ez-greedy':
        pass
    elif exp_type == 'QMP':
        pass
    elif exp_type == 'levy':
        pass

    logger.info('initialize wandb')
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        id=datetime.now().strftime('%Y%m%d%H%M%S'),
        config=OmegaConf.to_container(cfg),
    )

    logger.info('start training')
    obs, info = vec_env.reset()  # environments are automatically reset
    success_rate = np.zeros(vec_env.num_envs)  # exponentially decayed success ratio, debug
    for epoch in range(1, num_epochs + 1):
        for step in trange(1, train_steps + 1, desc=f'epoch {epoch}', unit='step'):
            wandb_log = {'step': step + (epoch - 1) * train_steps}

            # advance environment
            if step > init_steps or epoch != 1:
                act = [mtmhsac.get_action(obs[i], i) for i in range(vec_env.num_envs)]
            else:
                act = vec_env.sample_action()
            obs, rwd, ter, tru, info = vec_env.step(act)

            # update training success ratio
            for i in range(vec_env.num_envs):
                if ter[i] or tru[i]:
                    success_rate[i] = success_rate[i] * (1 - sr_decay) + info[i]['success'] * sr_decay
            wandb_log.update({'train/success-rate': success_rate.mean()})

            # store transitions
            for i in range(vec_env.num_envs):
                buffer.append(obs[i], act[i], rwd[i], info[i]['next_observation'], ter[i], i)

            # fit
            if step > init_steps or epoch != 1:
                # fit once every vec_env.num_envs (environment) steps
                fit_log = mtmhsac.update(buffer.sample(batch_size))
                wandb_log.update(fit_log)

            # log to wandb
            if step % log_interval == 0:
                run.log(wandb_log)

        eval_log = evaluate(mtmhsac, eval_envs, num_episodes=eval_episodes)
        # logger.info(eval_log)  # do I really need to log this? not actually...?
        # convert into wandb-friendly format
        eval_log = {key: value.mean() for key, value in eval_log.items()}
        eval_log['epoch'] = epoch
        run.log(eval_log)

    vec_env.close()
    for env in eval_envs: env.close()
    run.finish()


if __name__ == '__main__':
    main()