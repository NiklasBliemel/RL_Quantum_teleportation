import torch.nn as nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class ActionModel(nn.Module):
    def __init__(self, max_steps, n_actions):
        super(ActionModel, self).__init__()
        self.w = nn.Parameter(
            torch.zeros(max_steps, n_actions * 3, n_actions, dtype=torch.float32, requires_grad=True))
        self.start = nn.Parameter(torch.zeros(n_actions, dtype=torch.float32, requires_grad=True))
        self.index_range = torch.arange(max_steps)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        out = observation.clone().detach() if torch.is_tensor(observation) else torch.tensor(observation)
        return self.w[self.index_range[out != -1].int(), out[out != -1].int()].sum(axis=0) + self.start
