from typing import Tuple
from math import exp, log
import numpy as np
import torch
from algos import SAC
import imageio


__all__ = ['compute_q', 'compute_entropy_gaussian']


@torch.no_grad()
def compute_q(sac: SAC, transition: Tuple[np.ndarray, ...]) -> Tuple[float, float]:
    obs, act, rwd, nobs, done = map(sac._tensor, transition)
    rwd = rwd.unsqueeze(-1)
    done = done.unsqueeze(-1)

    # compute target Q values
    nact, nlogp = sac._get_action(nobs)
    nq1 = sac.target1(torch.cat([nobs, nact], dim=-1))
    nq2 = sac.target2(torch.cat([nobs, nact], dim=-1))
    nq = torch.minimum(nq1, nq2) - exp(sac.log_temp) * nlogp
    td_target = rwd + (1 - done) * sac.gamma * nq

    # compute Q values
    q1 = sac.critic1(torch.cat([obs, act], dim=-1))
    q2 = sac.critic2(torch.cat([obs, act], dim=-1))
    q = torch.minimum(q1, q2)

    return q.item(), td_target.item()

def compute_entropy_gaussian(log_std: np.ndarray) -> np.ndarray:
    # log_std is a 1D array of log standard deviations of a multi-dimensional Gaussian distribution
    d = len(log_std)
    return 0.5 * d * (1 + log(2 * np.pi)) + 0.5 * np.sum(log_std)
