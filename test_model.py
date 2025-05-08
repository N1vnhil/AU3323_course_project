import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(53510713690200)


class Actor(nn.Module):
    def __init__(self, state_dim=12, action_dim=4):
        super(Actor, self).__init__()

        # 简单的前馈网络，没有归一化层
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # 输出范围[-1,1]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(next(self.parameters()).device)
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim=12):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        return self.net(state)


class RunningNormalize:
    def __init__(self, shape, clip=5.0):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.clip = clip
        self.running_mean = np.zeros(shape, dtype=np.float32)
        self.running_var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4
        self.training = True
        self.momentum = 0.95

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def reset(self):
        self.running_mean = np.zeros(self.shape, dtype=np.float32)
        self.running_var = np.ones(self.shape, dtype=np.float32)
        self.count = 1e-4

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)

        original_shape = x.shape
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if self.training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]

            self.running_mean = (self.momentum * self.running_mean +
                                 (1 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var +
                                (1 - self.momentum) * batch_var)
            self.count += batch_count

        x_normalized = ((x - self.running_mean) /
                        (np.sqrt(self.running_var + 1e-8)))
        x_clipped = np.clip(x_normalized, -self.clip, self.clip)

        if len(original_shape) == 1:
            x_clipped = x_clipped.squeeze(0)

        return x_clipped

    @property
    def mean(self):
        return self.running_mean

    @property
    def var(self):
        return self.running_var