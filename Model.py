import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(53510713690200)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = torch.tanh(x)
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.l1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.l2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 1)
        self.q1_dropout = nn.Dropout(0.05)

        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.l4 = nn.LayerNorm(256, 128)
        self.fc5 = nn.Linear(256, 128)
        self.l5 = nn.LayerNorm(128)
        self.l6 = nn.Linear(128, 1)
        self.q2_dropout = nn.Dropout(0.05)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = self.l1(self.fc1(sa))
        q1 = F.relu(q1)
        q1 = self.q1_dropout(q1)
        q1 = self.l2(self.fc2(q1))
        q1 = F.relu(q1)
        q1 = self.fc3(q1)

        # Q2
        q2 = self.l4(self.fc4(sa))
        q2 = F.relu(q2)
        q2 = self.q2_dropout(q2)
        q2 = self.l5(self.fc5(q2))
        q2 = F.relu(q2)
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.l1(self.fc1(sa))
        q1 = F.relu(q1)
        q1 = self.q1_dropout(q1)
        q1 = self.l2(self.fc2(q1))
        q1 = F.relu(q1)
        q1 = self.fc3(q1)
        return q1