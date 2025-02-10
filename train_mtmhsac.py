from typing import Union

import torch
import gymnasium as gym
import metaworld
import hydra
from omegaconf import DictConfig
from tqdm import trange

from envs import generate_metaworld_env_fns, SubprocVecEnv
from algos import MTMHSAC
from buffers import ReplayBuffer


def parse_benchmark(benchmark: Union[list[str], str], seed: int = None) -> tuple[SubprocVecEnv, list[gym.Env], int, int]:
    '''Can be given a list of environment names or a benchmark name.
    Return a SubprocVecEnv, a list of gym environments, observation dimension, and action dimension.
    SubprocVecEnv is a vectorized environment that runs multiple environments in parallel.
    A list of gym environments is used for evaluation.
    '''
    if isinstance(benchmark, list):
        env_fns = generate_metaworld_env_fns(benchmark, seed=seed)
        obs_dim, act_dim = 39, 4
    elif isinstance(benchmark, str):
        if benchmark == 'MT10':
            mt10 = metaworld.MT10(seed=seed)
            env_fns = generate_metaworld_env_fns(mt10, seed=seed)
            obs_dim, act_dim = 39, 4
        elif benchmark == 'MT50':
            mt50 = metaworld.MT50(seed=seed)
            env_fns = generate_metaworld_env_fns(mt50, seed=seed)
            obs_dim, act_dim = 39, 4
        else:
            raise ValueError(f'Invalid benchmark name: {benchmark}')
    else:
        raise TypeError(f'Invalid benchmark type: {type(benchmark)}')
    return SubprocVecEnv(env_fns), [env_fn() for env_fn in env_fns], obs_dim, act_dim

def evaluate(mtmhsac: MTMHSAC, eval_envs: list[gym.Env], num_episodes: int = 10):
    mean_return = []
    for i, env in enumerate(eval_envs):
        total_rwd = 0
        episode_cnt = 0
        obs, _ = env.reset()
        while episode_cnt < num_episodes:
            act = mtmhsac.get_action(obs, i, deterministic=True)
            obs, rwd, ter, tru, _ = env.step(act)
            total_rwd += rwd
            if ter or tru: episode_cnt += 1
        mean_return.append(total_rwd / num_episodes)
    return mean_return

@hydra.main(version_base=None, config_path='configs', config_name='train_mtmhsac')
def main(cfg: DictConfig) -> None:
    seed: int = cfg.seed
    gpu_index: int = cfg.gpu_index
    device = f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu'
    benchmark: Union[list[str], str] = cfg.benchmark

    train_steps: int = cfg.train_steps  # steps are counted for each environment
    init_steps: int = cfg.init_steps
    eval_freq: int = cfg.eval_freq
    buffer_size: int = cfg.buffer_size
    batch_size: int = cfg.batch_size

    vec_env, eval_envs, obs_dim, act_dim = parse_benchmark(benchmark, seed=seed)
    mtmhsac = MTMHSAC(vec_env.num_envs, obs_dim, act_dim, **cfg.mtmhsac, device=device)
    buffer = ReplayBuffer(buffer_size, seed=seed)

    obs, info = vec_env.reset()  # environments are automatically reset
    for step in trange(1, train_steps + 1):
        # advance environment
        if step <= init_steps:
            act = vec_env.sample_action()
        else:
            act = [mtmhsac.get_action(obs[i], i) for i in range(vec_env.num_envs)]
        obs, rwd, ter, _, info = vec_env.step(act)

        # store transitions
        for i in range(vec_env.num_envs):
            buffer.append(obs[i], act[i], rwd[i], info[i]['next_observation'], ter[i], i)

        # fit
        if step > init_steps:
            for _ in range(vec_env.num_envs):
                log = mtmhsac.update(buffer.sample(batch_size))

        # evaluate
        if step % eval_freq == 0:
            print(evaluate(mtmhsac, eval_envs))


if __name__ == '__main__':
    main()