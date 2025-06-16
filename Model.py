import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.norm3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.relu(self.norm3(self.fc3(x)))  # 增加的隐藏层激活函数
        x = torch.tanh(self.fc4(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_norm1 = nn.LayerNorm(256)
        self.q1_fc2 = nn.Linear(256, 128)
        self.q1_norm2 = nn.LayerNorm(128)
        self.q1_fc3 = nn.Linear(128, 64)  # 增加的隐藏层
        self.q1_norm3 = nn.LayerNorm(64)  # 增加的归一化层
        self.q1_fc4 = nn.Linear(64, 1)

        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_norm1 = nn.LayerNorm(256)
        self.q2_fc2 = nn.Linear(256, 128)
        self.q2_norm2 = nn.LayerNorm(128)
        self.q2_fc3 = nn.Linear(128, 64)  # 增加的隐藏层
        self.q2_norm3 = nn.LayerNorm(64)  # 增加的归一化层
        self.q2_fc4 = nn.Linear(64, 1)

    def forward(self, state, action):
        q1_input = torch.cat([state, action], dim=1)  # 合并状态和动作
        q1 = F.relu(self.q1_norm1(self.q1_fc1(q1_input)))
        q1 = F.relu(self.q1_norm2(self.q1_fc2(q1)))
        q1 = F.relu(self.q1_norm3(self.q1_fc3(q1)))  # 增加的隐藏层激活函数
        q1 = self.q1_fc4(q1)
        
        q2_input = torch.cat([state, action], dim=1)  # 合并状态和动作
        q2 = F.relu(self.q2_norm1(self.q2_fc1(q2_input)))
        q2 = F.relu(self.q2_norm2(self.q2_fc2(q2)))
        q2 = F.relu(self.q2_norm3(self.q2_fc3(q2)))  # 增加的隐藏层激活函数
        q2 = self.q2_fc4(q2)
        
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_norm1(self.q1_fc1(sa)))
        q1 = F.relu(self.q1_norm2(self.q1_fc2(q1)))
        q1 = self.q1_fc3(q1)
        q1 = self.q1_fc4(q1)
        return q1
