"""
Deep Q-Network for Music Recommendation.
"""

import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    """Neural network for Q-value estimation."""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        return self.layers(state)


if __name__ == "__main__":
    # Quick test
    net = QNetwork(9, 100)
    x = torch.randn(1, 9)
    out = net(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print("QNetwork test passed!")
