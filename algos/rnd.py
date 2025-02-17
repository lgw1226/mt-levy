import torch
import numpy as np
from components import MLP


class RunningMeanStd():
    def __init__(self, shape: tuple[int], device: torch.device = torch.device('cpu')):
        self.n = 0
        self.mu = torch.zeros(shape, dtype=torch.float32, device=device)
        self.m2 = torch.zeros(shape, dtype=torch.float32, device=device)

    def update(self, x: torch.Tensor):
        self.n += 1
        delta = x - self.mu
        self.mu += delta / self.n
        delta2 = x - self.mu
        self.m2 += delta * delta2

    def mean(self) -> torch.Tensor:
        return self.mu
    
    def variance(self) -> torch.Tensor:
        return self.m2 / (self.n - 1) if self.n > 1 else torch.zeros_like(self.m2) + 1e-3
    
    def stddev(self) -> torch.Tensor:
        return self.variance().sqrt()


class RND():
    def __init__(
            self, 
            obs_dim: int, 
            embed_dim: int = 64, 
            hidden_layers: list[int] = [256, 256], 
            lr: float = 1e-4,
            device: torch.device = torch.device('cpu')
    ):
        super(RND, self).__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.device = device
        self.predictor = MLP([obs_dim] + hidden_layers + [embed_dim]).to(device)
        self.target = MLP([obs_dim] + hidden_layers + [embed_dim]).to(device).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self._reset()

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs: np.ndarray) -> float:
        # convert to tensor
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # compute intrinsic reward
        target = self.target(obs)
        predictor = self.predictor(obs)
        intrinsic_reward = (target - predictor).pow(2).sum(dim=-1).sqrt()

        # normliaze intrinsic reward and update statistics
        intrinsic_reward = (intrinsic_reward - self.rwd_stats.mean()) / self.rwd_stats.stddev()
        self.rwd_stats.update(intrinsic_reward)

        return intrinsic_reward.item()
    
    def update(self, batch_obs: np.ndarray) -> float:
        # normalize batch observations
        batch_obs = torch.tensor(batch_obs, dtype=torch.float32, device=self.device)

        # compute loss and update predictor
        target = self.target(batch_obs)
        predictor = self.predictor(batch_obs)
        loss = (target - predictor).pow(2).sum(dim=-1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _reset(self):
        self.rwd_stats = RunningMeanStd((1,), device=self.device)


class IntrinsicValue():

    def __init__(self, obs_dim: int, hidden_layers: list[int] = [256, 256], lr: float = 1e-4):
        self.obs_dim = obs_dim
        self.nn = MLP([obs_dim] + hidden_layers + [1])
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=1e-4)
