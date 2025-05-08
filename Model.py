import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(53510713690200)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)  # 增加dropout率

        # Skip connection handling
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.linear1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.ln2(out)
        out = self.dropout(out)  # 添加额外的dropout

        out += identity
        out = self.activation(out)
        return out


class Actor(nn.Module):
    def __init__(self, state_dim=12, action_dim=4):
        super(Actor, self).__init__()

        # 使用LayerNorm替代BatchNorm
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.float()
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim=12):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.float()
        return self.net(state)


class RunningNormalize:
    def __init__(self, shape, clip=10.0):  # 增大clip范围
        self.shape = shape
        self.clip = clip
        self.running_mean = np.zeros(shape)
        self.running_var = np.ones(shape)
        self.count = 1e-4
        self.training = True
        self.momentum = 0.99  # 添加动量参数

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def reset(self):
        self.running_mean = np.zeros(self.shape)
        self.running_var = np.ones(self.shape)
        self.count = 1e-4

    def __call__(self, x):
        x = np.array(x, dtype=np.float32)

        original_shape = x.shape
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if self.training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]

            # 使用动量更新
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            self.count += batch_count

        # 标准化
        x_normalized = (x - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)
        x_clipped = np.clip(x_normalized, -self.clip, self.clip)

        if original_shape == x_clipped.shape[1:]:
            return x_clipped.squeeze()
        return x_clipped

    @property
    def mean(self):
        return self.running_mean

    @property
    def var(self):
        return self.running_var