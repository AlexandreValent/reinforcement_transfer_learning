"""
Neural network module.

This module defines architectures used by reinforcement learning agents.
"""

import torch
import torch.nn
from rl_lunarlanding.config import CFG

class DQN(torch.nn.Module):
    """
    PyTorch implementation of a Deep Q-Network.
    x_dim refers to the number of dimensions to pass as input.
    y_dim refers to the action space of the agent.
    """

    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(

            torch.nn.Linear(CFG.x_dim, 16),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(16, 8),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(8, CFG.y_dim),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, obs):
        return self.net(obs)
