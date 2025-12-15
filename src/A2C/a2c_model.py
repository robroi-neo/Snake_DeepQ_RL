# a2c_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ActorCritic(nn.Module):
    def __init__(self, state_size=19, hidden=128, action_size=3):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden)

        # actor
        self.actor = nn.Linear(hidden, action_size)

        # critic
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value
