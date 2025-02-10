from copy import deepcopy
import torch
import torch.nn as nn
from models import create_mlp_layers


class DoubleCritic(nn.Module):

    def __init__(self, critic_arch: list[int], bound_output: bool = False, weight_gain: float = 1.0):
        super(DoubleCritic, self).__init__()
        self.layers = create_mlp_layers(critic_arch)
        if bound_output:
            self.layers.append(nn.Sigmoid())

        def _init_weights(ml: nn.ModuleList) -> nn.ModuleList:
            for m in ml:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=weight_gain)
                    nn.init.zeros_(m.bias)
            return ml
        
        self.critic1 = nn.Sequential(*_init_weights(deepcopy(self.layers)))
        self.target1 = deepcopy(self.critic1).requires_grad_(False)
        self.critic2 = nn.Sequential(*_init_weights(deepcopy(self.layers)))
        self.target2 = deepcopy(self.critic2).requires_grad_(False)

    def forward(self, observation: torch.Tensor, action: torch.Tensor, use_targets: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([observation, action], dim=-1)
        if use_targets:
            return self.target1(x), self.target2(x)
        else:
            return self.critic1(x), self.critic2(x)
        
    def update_targets(self, tau: float):
        for target, critic in zip([self.target1, self.target2], [self.critic1, self.critic2]):
            tsd = target.state_dict()
            csd = critic.state_dict()
            for key in csd.keys():
                tsd[key] = tau * csd[key] + (1 - tau) * tsd[key]
            target.load_state_dict(tsd)
        
    def __call__(self, observation: torch.Tensor, action: torch.Tensor, use_targets: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(observation, action, use_targets=use_targets)
