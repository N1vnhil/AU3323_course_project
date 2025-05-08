import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model import Actor, Critic, RunningNormalize
from collections import deque
import random
from datetime import datetime
import os


torch.manual_seed(53510713690200)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.buffer)

    def push(self, state, next_state, action, reward, done):
        self.buffer.append((state, next_state, action, reward, done))

    def sample(self, batch_size, device):
        sample = random.sample(range(len(self.buffer)), batch_size)
        state = torch.from_numpy(np.vstack([self.buffer[i][0] for i in sample])).float().to(device)
        next_state = torch.from_numpy(np.vstack([self.buffer[i][1] for i in sample])).float().to(device)
        action = torch.from_numpy(np.vstack([self.buffer[i][2] for i in sample])).float().to(device)
        reward = torch.from_numpy(np.vstack([self.buffer[i][3] for i in sample])).float().to(device)
        done = torch.from_numpy(np.vstack([self.buffer[i][4] for i in sample])).float().to(device)
        return state, next_state, action, reward, done


class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=5e-4, gamma=0.995,
                 epsilon=0.2, epochs=10, gae_lambda=0.98, entropy_coef=0.01,
                 value_clip=0.2, max_grad_norm=1.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        # Optimizers
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=critic_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Learning rate schedulers with warmup
        self.actor_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.actor_opt,
            max_lr=actor_lr,
            total_steps=10000,
            pct_start=0.1,
            final_div_factor=1e4
        )
        self.critic_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.critic_opt,
            max_lr=critic_lr,
            total_steps=10000,
            pct_start=0.1,
            final_div_factor=1e4
        )

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_clip = value_clip
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = 0.5
        self.max_kl = 0.015
        self.target_kl = 0.01

        # Action space parameters
        self.action_high = np.array([1.0, np.pi / 4, np.pi / 4, np.pi / 4])
        self.action_low = np.array([0.0, -np.pi / 4, -np.pi / 4, -np.pi / 4])

        # Normalizers
        self.state_normalizer = RunningNormalize(state_dim, clip=20.0)
        self.reward_normalizer = RunningNormalize(1, clip=10.0)

        # Action noise parameters
        self.action_std = 1.0
        self.action_std_decay = 0.9999
        self.min_action_std = 0.1

        # Replay buffer
        self.replay_buffer = ReplayBuffer(100000)
        self.min_buffer_size = 1000
        self.replay_batch_size = 256

    def train(self):
        self.actor.train()
        self.critic.train()
        self.state_normalizer.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.state_normalizer.eval()

    def compute_gae(self, rewards, values, next_values, dones):
        # 根据环境奖励范围标准化
        rewards = rewards / 10000.0
        values = values / 10000.0
        next_values = next_values / 10000.0

        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def get_action(self, state):
        state = self.state_normalizer(state)

        if isinstance(state, np.ndarray):
            if state.ndim == 1:
                state = state.reshape(1, -1)
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            if state.ndim == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)

        with torch.no_grad():
            action_mean = self.actor(state)
            action_std = torch.ones_like(action_mean) * self.action_std
            dist = torch.distributions.Normal(action_mean, action_std)

            # Sample and clip action
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
            log_prob = dist.log_prob(action)

            # Decay action noise
            self.action_std = max(self.min_action_std,
                                  self.action_std * self.action_std_decay)

        # Scale actions to environment space
        action = action.cpu().numpy()
        if action.shape[0] == 1:
            action = action.squeeze(0)

        # 分别处理推力和角度
        scaled_action = np.zeros_like(action)
        scaled_action[..., 0] = (action[..., 0] + 1.0) * 0.5  # 推力映射到[0,1]
        scaled_action[..., 1:] = action[..., 1:] * (np.pi / 4)  # 角度映射到[-pi/4, pi/4]

        # 确保动作在有效范围内
        scaled_action = np.clip(scaled_action, self.action_low, self.action_high)

        return scaled_action, log_prob.cpu().numpy()

    def update(self, states, actions, rewards, next_states, dones, old_log_probs):
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        # Mix with replay buffer data if available
        if len(self.replay_buffer) > self.min_buffer_size:
            replay_data = self.replay_buffer.sample(
                min(self.replay_batch_size, len(self.replay_buffer)),
                self.device
            )
            states = torch.cat([states, replay_data[0]], dim=0)
            next_states = torch.cat([next_states, replay_data[1]], dim=0)
            actions = torch.cat([actions, replay_data[2]], dim=0)
            rewards = torch.cat([rewards, replay_data[3]], dim=0)
            dones = torch.cat([dones, replay_data[4]], dim=0)

        # Compute value estimates
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Mini-batch training
        batch_size = states.shape[0]
        mini_batch_size = 512
        accumulation_steps = 2

        for epoch in range(self.epochs):
            indices = torch.randperm(batch_size)

            total_actor_loss = 0
            total_critic_loss = 0
            total_kl_div = 0

            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()

            for start_idx in range(0, batch_size, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]

                # Critic update
                for _ in range(2):
                    values_pred = self.critic(mb_states)
                    values_clipped = values[mb_indices] + torch.clamp(
                        values_pred - values[mb_indices],
                        -self.value_clip,
                        self.value_clip
                    )
                    critic_loss1 = F.mse_loss(values_pred, mb_returns)
                    critic_loss2 = F.mse_loss(values_clipped, mb_returns)
                    critic_loss = self.value_loss_coef * torch.max(critic_loss1, critic_loss2)

                    (critic_loss / accumulation_steps).backward()
                    total_critic_loss += critic_loss.item()

                # Actor update
                action_mean = self.actor(mb_states)
                action_std = torch.ones_like(action_mean) * self.action_std
                dist = torch.distributions.Normal(action_mean, action_std)

                curr_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO objectives
                ratio = torch.exp(curr_log_probs.sum(-1, keepdim=True) - mb_old_log_probs.sum(-1, keepdim=True))

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages
                actor_loss = -(torch.min(surr1, surr2).mean() + self.entropy_coef * entropy)

                (actor_loss / accumulation_steps).backward()
                total_actor_loss += actor_loss.item()

                # Calculate KL divergence
                approx_kl_div = ((mb_old_log_probs - curr_log_probs) ** 2).mean()
                total_kl_div += approx_kl_div.item()

            # Average KL divergence
            avg_kl_div = total_kl_div / (batch_size // mini_batch_size)

            # Early stopping based on KL divergence
            if avg_kl_div > self.max_kl:
                break

            # Apply accumulated gradients
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm
            )

            self.actor_opt.step()
            self.critic_opt.step()

            # Update learning rates
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        # Store transition in replay buffer
        self.replay_buffer.push(
            states.cpu().numpy(),
            next_states.cpu().numpy(),
            actions.cpu().numpy(),
            rewards.cpu().numpy(),
            dones.cpu().numpy()
        )

        return (total_critic_loss / (batch_size // mini_batch_size),
                total_actor_loss / (batch_size // mini_batch_size))

    def save(self, path=None):
        if path is None:
            path = f'checkpoints/ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_opt.state_dict(),
            'critic_optimizer_state_dict': self.critic_opt.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
            'state_normalizer_state': {
                'mean': self.state_normalizer.mean,
                'var': self.state_normalizer.var,
                'count': self.state_normalizer.count
            },
            'reward_normalizer_state': {
                'mean': self.reward_normalizer.mean,
                'var': self.reward_normalizer.var,
                'count': self.reward_normalizer.count
            }
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        actor_state_dict = {k.replace("module.", ""): v
                            for k, v in checkpoint['actor_state_dict'].items()}
        critic_state_dict = {k.replace("module.", ""): v
                             for k, v in checkpoint['critic_state_dict'].items()}

        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

        self.actor_opt.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])

        state_normalizer_state = checkpoint['state_normalizer_state']
        self.state_normalizer.running_mean = state_normalizer_state['mean']
        self.state_normalizer.running_var = state_normalizer_state['var']
        self.state_normalizer.count = state_normalizer_state['count']

        reward_normalizer_state = checkpoint['reward_normalizer_state']
        self.reward_normalizer.running_mean = reward_normalizer_state['mean']
        self.reward_normalizer.running_var = reward_normalizer_state['var']
        self.reward_normalizer.count = reward_normalizer_state['count']

