import torch
import torch.nn.functional as F
import numpy as np
from test_model import Actor, Critic, RunningNormalize
from collections import deque
import random
import os

torch.manual_seed(53510713690200)


class PPOAgent:
    def __init__(self, state_dim=12, action_dim=4, actor_lr=3e-4, critic_lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        # 调整超参数
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epochs = 5
        self.gae_lambda = 0.95
        self.entropy_coef = 0.001
        self.max_grad_norm = 0.5

        # 减小动作噪声
        self.action_std = 0.1
        self.action_std_decay = 0.9999
        self.min_action_std = 0.05

        # 优化器使用较小的学习率
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        # Action space parameters
        self.action_high = np.array([1.0, np.pi / 4, np.pi / 4, np.pi / 4])
        self.action_low = np.array([0.0, -np.pi / 4, -np.pi / 4, -np.pi / 4])

        # State normalization
        self.state_normalizer = RunningNormalize(state_dim)

    def get_action(self, state):
        """获取动作和对应的log概率"""
        # 规范化状态
        state = self.state_normalizer(state)

        # 转换为tensor
        if isinstance(state, np.ndarray):
            if state.ndim == 1:
                state = state.reshape(1, -1)
            state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            # 获取动作分布
            action_mean = self.actor(state)
            action_std = torch.ones_like(action_mean) * self.action_std
            dist = torch.distributions.Normal(action_mean, action_std)

            # 采样动作
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
            log_prob = dist.log_prob(action).sum(-1)

            # 转换为numpy
            action = action.cpu().numpy()
            log_prob = log_prob.cpu().numpy()

            # 缩放动作到环境范围
            scaled_action = np.zeros_like(action)
            scaled_action[..., 0] = (action[..., 0] + 1.0) * 0.5  # thrust [0, 1]
            scaled_action[..., 1:] = action[..., 1:] * (np.pi / 4)  # angles [-pi/4, pi/4]

            # 确保动作在有效范围内
            scaled_action = np.clip(scaled_action, self.action_low, self.action_high)

            if scaled_action.shape[0] == 1:
                scaled_action = scaled_action.squeeze(0)
                log_prob = log_prob.squeeze(0)

            return scaled_action, log_prob

    def compute_gae(self, rewards, values, next_values, dones):
        """计算广义优势估计"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_value = next_values[t] if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, rewards, next_states, dones, old_log_probs):
        """更新策略和价值网络"""
        # 归一化奖励
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # 计算优势和回报
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)
            # 归一化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        for _ in range(self.epochs):
            # Actor更新
            action_mean = self.actor(states)
            # 确保action_std为正数
            action_std = torch.ones_like(action_mean).to(self.device) * max(self.action_std, 1e-6)

            try:
                dist = torch.distributions.Normal(action_mean, action_std)
                curr_log_probs = dist.log_prob(actions).sum(-1)
                entropy = dist.entropy().mean()

                ratio = torch.exp(curr_log_probs - old_log_probs.sum(-1))
                # 裁剪ratio以防止数值不稳定
                ratio = torch.clamp(ratio, 1e-10, 10.0)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

                actor_loss = -(torch.min(surr1, surr2).mean() + self.entropy_coef * entropy)

                # Critic更新
                value_pred = self.critic(states)
                critic_loss = F.mse_loss(value_pred, returns)

                # 检查损失值是否为NaN
                if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                    print("NaN loss detected, skipping update")
                    continue

                # 更新网络
                self.actor_opt.zero_grad()
                actor_loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

            except RuntimeError as e:
                print(f"Error during update: {e}")
                continue

            return critic_loss.item(), actor_loss.item()

        return 0.0, 0.0  # 如果所有更新都失败，返回零损失

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'state_normalizer_state': {
                'mean': self.state_normalizer.mean,
                'var': self.state_normalizer.var,
                'count': self.state_normalizer.count
            }
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        state_normalizer_state = checkpoint['state_normalizer_state']
        self.state_normalizer.running_mean = state_normalizer_state['mean']
        self.state_normalizer.running_var = state_normalizer_state['var']
        self.state_normalizer.count = state_normalizer_state['count']

    def eval(self):
        """设置为评估模式"""
        self.actor.eval()
        self.critic.eval()
        self.state_normalizer.eval()

    def train(self):
        """设置为训练模式"""
        self.actor.train()
        self.critic.train()
        self.state_normalizer.train()