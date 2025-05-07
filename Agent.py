import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model import Actor, Critic
from collections import deque
import random
from datetime import datetime


torch.manual_seed(53510713690200)


class ReplayBuffer():
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
     
class DDPGAgent():
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, buffer_size, batch_size, gamma, tau):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_opt =  torch.optim.Adam(self.actor.parameters(), actor_lr)
            
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)        
        self.critic_opt =  torch.optim.Adam(self.critic.parameters(), critic_lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        # start of your code
        # This method returns actions the agent output during the training process 
        
        action = None
        # end of your code   
        return action
    
    def update(self, state, next_state, action, reward, done):
        self.buffer.push(state, next_state, action, reward, done)
        if len(self.buffer) < self.batch_size:
            return None, None
        state, next_state, action, reward, done = self.buffer.sample(self.batch_size, self.device)
        # start of your code
        # You need to compute actor_loss/critic_loss and implement the backpropagation in this block
        # function you may need: self.actor.zero_grad(), self.actor_opt.step(), actor_loss.backward()

        critic_loss, actor_loss = None, None
        
        # end of your code
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau*param + (1-self.tau)*target_param)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau*param + (1-self.tau)*target_param)
        return critic_loss, actor_loss 
    
    def save(self):
        torch.save(self.actor.state_dict(), 'checkpoints/actor.pth')
        torch.save(self.critic.state_dict(), 'checkpoints/critic.pth')
        torch.save(self.actor_target.state_dict(), 'checkpoints/actor_target.pth')
        torch.save(self.critic_target.state_dict(), 'checkpoints/critic_target.pth')

    def load(self, actor, critic, actor_target, critic_target):
        self.actor.load_state_dict(torch.load(actor, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic, map_location=self.device))
        self.actor_target.load_state_dict(torch.load(actor_target, map_location=self.device))
        self.critic_target.load_state_dict(torch.load(critic_target, map_location=self.device))


class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, epsilon=0.2, epochs=10):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), critic_lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs

        # Action bounds from environment
        self.action_high = np.array([1, np.pi / 4, np.pi / 4, np.pi / 4])
        self.action_low = np.array([0, -np.pi / 4, -np.pi / 4, -np.pi / 4])

    def get_action(self, state):
        # Handle both numpy array and tensor inputs
        if isinstance(state, np.ndarray):
            # Handle both single state and batch of states
            if state.ndim == 1:
                state = state.reshape(1, -1)
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            if state.ndim == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)

        with torch.no_grad():
            action_mean = self.actor(state)
            action_std = torch.ones_like(action_mean) * 0.1
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Convert from [-1,1] to actual action space
        action = action.cpu().numpy()
        if action.shape[0] == 1:
            action = action.squeeze(0)
        scaled_action = np.zeros_like(action)
        for i in range(len(self.action_high)):
            scaled_action[..., i] = (action[..., i] + 1) * (self.action_high[i] - self.action_low[i]) / 2 + \
                                    self.action_low[i]

        return scaled_action, log_prob.cpu().numpy()

    def update(self, states, actions, rewards, next_states, dones, old_log_probs):
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        # Compute value estimates
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

            # Compute returns and advantages
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            # Get current action distributions
            action_mean = self.actor(states)
            action_std = torch.ones_like(action_mean) * 0.1
            dist = torch.distributions.Normal(action_mean, action_std)
            curr_log_probs = dist.log_prob(actions)

            # Sum log probs across action dimensions
            curr_log_probs_sum = curr_log_probs.sum(-1, keepdim=True)
            old_log_probs_sum = old_log_probs.sum(-1, keepdim=True)

            # Compute ratio and surrogate losses
            ratio = torch.exp(curr_log_probs_sum - old_log_probs_sum)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(self.critic(states), returns)

            # Update networks
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_opt.step()

        return critic_loss.item(), actor_loss.item()

    def save(self):
        torch.save(self.actor.state_dict(), f'checkpoints/ppo_actor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        torch.save(self.critic.state_dict(), f'checkpoints/ppo_critic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')

    def load(self, actor_path, critic_path):
        actor_state_dict = torch.load(actor_path, map_location=self.device)
        actor_state_dict = {k.replace("module.", ""): v for k, v in actor_state_dict.items()}
        self.actor.load_state_dict(actor_state_dict)

        critic_state_dict = torch.load(critic_path, map_location=self.device)
        critic_state_dict = {k.replace("module.", ""): v for k, v in critic_state_dict.items()}
        self.critic.load_state_dict(critic_state_dict)


