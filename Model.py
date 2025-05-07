import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(53510713690200)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, action_dim)

        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        self.ln4 = nn.LayerNorm(64)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        x = F.relu(self.ln3(self.l3(x)))
        x = F.relu(self.ln4(self.l4(x)))
        x = torch.tanh(self.l5(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):  # action_dim kept for compatibility
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 384)
        self.l3 = nn.Linear(384, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 1)

        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(384)
        self.ln3 = nn.LayerNorm(256)
        self.ln4 = nn.LayerNorm(128)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state):
        x = F.relu(self.ln1(self.l1(state)))
        x = F.relu(self.ln2(self.l2(x)))
        x = F.relu(self.ln3(self.l3(x)))
        x = F.relu(self.ln4(self.l4(x)))
        x = self.l5(x)
        return x