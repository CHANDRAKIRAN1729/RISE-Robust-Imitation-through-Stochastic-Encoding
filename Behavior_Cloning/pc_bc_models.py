#!/usr/bin/env python3
"""
Parameter-Conditioned Behavior Cloning (PC-BC) models.
Defines a PolicyNetwork that directly consumes (state s, goal g, obstacle parameters c)
concatenated as input and predicts continuous action (v, w).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Two-layer MLP with 128 hidden units per layer.

    Inputs:
      - state:  (B, state_dim)
      - goal:   (B, goal_dim)
      - cond c: (B, cond_dim)  # obstacle parameters: pos(x,y), vel(x,y), radius -> 5

    Output:
      - action: (B, action_dim)  # [v, w]
    """
    def __init__(
        self,
        state_dim: int = 3,
        goal_dim: int = 2,
        cond_dim: int = 5,
        action_dim: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = state_dim + goal_dim + cond_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

        # Optional: small init to avoid large outputs at start
        nn.init.uniform_(self.out.weight, -0.01, 0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, state: torch.Tensor, goal: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, goal, cond], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.out(h)
