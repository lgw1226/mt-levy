from typing import Union
import metaworld
from wrappers import MTWrapper


def generate_metaworld_env_fns(benchmark: Union[metaworld.Benchmark, list[str]], seed: int = None, **kwargs):
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
                base_env = base_env_cls(**kwargs)
                # return MTWrapper(base_env, base_env_tasks, sparse_reward=True, auto_reset=True, max_path_length=200, seed=seed)
                return MTWrapper(base_env, base_env_tasks, sparse_reward=False, auto_reset=True, max_path_length=200, seed=seed)  # debug
            env_fns.append(env_fn)
    elif isinstance(benchmark, metaworld.Benchmark):    
        for env_name in benchmark.train_classes:
            base_env_cls = benchmark.train_classes[env_name]
            base_env_tasks = []
            for task in benchmark.train_tasks:
                if task.env_name == env_name:
                    base_env_tasks.append(task)
            def env_fn():
                base_env = base_env_cls(**kwargs)
                # return MTWrapper(base_env, base_env_tasks, sparse_reward=True, auto_reset=True, max_path_length=200, seed=seed)
                return MTWrapper(base_env, base_env_tasks, sparse_reward=False, auto_reset=True, max_path_length=200, seed=seed)  # debug
            env_fns.append(env_fn)
    return env_fns


if __name__ == '__main__':

    seed = 0
    env_fns = generate_metaworld_env_fns(['reach-v2', 'push-v2'], seed=seed)

    from envs.subproc_vec_env import SubprocVecEnv
    env = SubprocVecEnv(env_fns)
    obs, info = env.reset()
    print(obs, info)

    for _ in range(201):
        act = env.sample_action()
        print(act)

        obs, rwd, ter, tru, info = env.step(act)
        print(obs, rwd, ter, tru, info)

    env.close()
