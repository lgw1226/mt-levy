from typing import Union
import metaworld
from wrappers import MTWrapper

import gymnasium as gym
from omegaconf import OmegaConf, ListConfig
from envs.subproc_vec_env import SubprocVecEnv



def generate_metaworld_env_fns(benchmark: Union[metaworld.Benchmark, list[str]], seed: int, **kwargs):
    """Generate a list of functions that return metaworld environments.

    :param benchmark: A metaworld benchmark or a list of environment names.
    :param seed: Random seed.
    :return: A list of functions that return metaworld environments.
    """

    env_fns = []

    if isinstance(benchmark, list):
        for env_name in benchmark:
            mt1 = metaworld.MT1(env_name, seed=seed)
            base_env_cls = mt1.train_classes[env_name]
            base_env_tasks = mt1.train_tasks
            def env_fn():
                base_env = base_env_cls()
                return MTWrapper(
                    base_env,
                    base_env_tasks,
                    sparse_reward=kwargs['sparse'],
                    auto_reset=True,
                    max_path_length=kwargs['horizon'],
                    seed=seed
                )
            env_fns.append(env_fn)

    elif isinstance(benchmark, metaworld.Benchmark):    
        for env_name in benchmark.train_classes:
            base_env_cls = benchmark.train_classes[env_name]
            base_env_tasks = []
            for task in benchmark.train_tasks:
                if task.env_name == env_name:
                    base_env_tasks.append(task)
            def env_fn():
                base_env = base_env_cls()
                return MTWrapper(
                    base_env,
                    base_env_tasks,
                    sparse_reward=kwargs['sparse'],
                    auto_reset=True,
                    max_path_length=kwargs['horizon'],
                    seed=seed
                )
            env_fns.append(env_fn)

    return env_fns

def parse_benchmark(benchmark: Union[list[str], str], seed: int, **kwargs) -> tuple[SubprocVecEnv, list[gym.Env], int, int]:
    '''Can be given a list of environment names or a benchmark name.
    Return a SubprocVecEnv, a list of gym environments, observation dimension, and action dimension.
    SubprocVecEnv is a vectorized environment that runs multiple environments in parallel.
    A list of gym environments is used for evaluation.
    '''

    if isinstance(benchmark, (list, ListConfig)):
        benchmark = OmegaConf.to_container(benchmark) if isinstance(benchmark, ListConfig) else benchmark
        env_fns = generate_metaworld_env_fns(benchmark, seed, **kwargs)
        obs_dim, act_dim = 39, 4
    elif isinstance(benchmark, str):
        if benchmark == 'MT10':
            mt10 = metaworld.MT10(seed=seed)
            env_fns = generate_metaworld_env_fns(mt10, seed, **kwargs)
            obs_dim, act_dim = 39, 4
        elif benchmark == 'MT50':
            mt50 = metaworld.MT50(seed=seed)
            env_fns = generate_metaworld_env_fns(mt50, seed, **kwargs)
            obs_dim, act_dim = 39, 4
        else:
            raise ValueError(f'Invalid benchmark name: {benchmark}')
    else:
        raise TypeError(f'Invalid benchmark type: {type(benchmark)}')
    return SubprocVecEnv(env_fns, seed), [env_fn() for env_fn in env_fns], obs_dim, act_dim
