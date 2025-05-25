import torch
import torch.nn.functional as F
import numpy as np
from test_model import Actor, Critic, RunningNormalize
from torch.distributions import Normal, TransformedDistribution, TanhTransform
import os

torch.manual_seed(53510713690200)


class PPOAgent:
    def __init__(self, state_dim=12, action_dim=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        # 调整超参数
        self.gamma = 0.99  # 保持适中的折扣因子
        self.gae_lambda = 0.95  # 保持适中的GAE参数
        self.epsilon = 0.2  # 保持PPO裁剪参数
        self.epochs = 10  # 保持训练轮数
        self.entropy_coef = 0.10
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        
        # 学习率衰减
        # self.actor_lr = 1e-4
        # self.critic_lr = 1e-4
        self.lr_decay = 0.999
        
        # 动作噪声
        self.action_std = 0.4  
        self.action_std_decay = 0.9999
        self.min_action_std = 0.1

        # 优化器使用较小的学习率
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # Action space parameters
        self.action_high = np.array([1.0, np.pi/2, np.pi/2, np.pi/2])
        self.action_low = np.array([0.0, -np.pi/2, -np.pi/2, -np.pi/2])

        # State normalization
        self.state_normalizer = RunningNormalize(state_dim)

    def get_action(self, state):
        state = self.state_normalizer(state)
        if isinstance(state, np.ndarray):
            if state.ndim == 1:
                state = state.reshape(1, -1)
            state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action_mean = self.actor(state)
            action_std = torch.ones_like(action_mean) * self.action_std
            base_dist = Normal(action_mean, action_std)
            dist = TransformedDistribution(base_dist, TanhTransform())
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

            action = action.cpu().numpy()
            log_prob = log_prob.cpu().numpy()

            # 保存原始动作值用于计算log_prob
            raw_action = action.copy()

            # 对动作进行缩放用于环境交互
            scaled_action = np.zeros_like(action)
            scaled_action[..., 0] = (action[..., 0] + 1.0) * 0.5
            scaled_action[..., 1:] = action[..., 1:] * (np.pi / 2)
            scaled_action = np.clip(scaled_action, self.action_low, self.action_high)

            if scaled_action.shape[0] == 1:
                scaled_action = scaled_action.squeeze(0)
                raw_action = raw_action.squeeze(0)
                log_prob = log_prob.squeeze(0)
            return scaled_action, log_prob, raw_action

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        returns = advantages + values
        return advantages, returns

    def compute_entropy(self, dist):
        """计算分布的熵"""
        # 对于TransformedDistribution，我们计算基础分布的熵
        if isinstance(dist, TransformedDistribution):
            return dist.base_dist.entropy().mean()
        return dist.entropy().mean()

    def update(self, states, actions, rewards, next_states, dones, old_log_probs):
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards = torch.clamp(rewards, -10, 10)  # 限制奖励范围
        
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            action_mean = self.actor(states)
            action_std = torch.ones_like(action_mean) * self.action_std
            base_dist = Normal(action_mean, action_std)
            dist = TransformedDistribution(base_dist, TanhTransform())
            curr_log_probs = dist.log_prob(actions).sum(-1)  # actions应该是[-1,1]范围内的原始动作值
            entropy = self.compute_entropy(dist)

            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -(torch.min(surr1, surr2).mean() + self.entropy_coef * entropy)

            value_pred = self.critic(states)
            critic_loss = F.mse_loss(value_pred, returns)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()

            kl_div = torch.mean((curr_log_probs - old_log_probs) ** 2)
            if kl_div > 0.02:
                break

            self.action_std = max(self.action_std * self.action_std_decay, self.min_action_std)
            for param_group in self.actor_opt.param_groups:
                param_group['lr'] *= self.lr_decay
            for param_group in self.critic_opt.param_groups:
                param_group['lr'] *= self.lr_decay

        return critic_loss.item(), actor_loss.item()

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