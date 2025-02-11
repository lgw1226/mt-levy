import metaworld
from wrappers import MTWrapper
from train_mtmhsac import parse_benchmark

seed = 1
# benchmark = 'MT10'
benchmark = ['reach-v2', 'drawer-close-v2', 'drawer-open-v2']
env, eval_envs, obs_dim, act_dim = parse_benchmark(benchmark, seed)

idx = 2

obs, info = env.reset()
for _ in range(201):
    act = env.sample_action()
    tup = env.step(act)
print(tup[0:2])
env.close()
