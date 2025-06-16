import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model import Actor, Critic
from collections import deque
import random

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
     
class TD3Agent():
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, buffer_size, batch_size, gamma, tau, policy_noise, noise_clip, policy_freq):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.actor_lr = actor_lr

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_opt =  torch.optim.Adam(self.actor.parameters(), self.actor_lr)
            
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)        
        self.critic_opt =  torch.optim.Adam(self.critic.parameters(), critic_lr, weight_decay=1e-4)

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def get_action(self, state, current_episode, max_episode):
        state = torch.from_numpy(state).float().to(self.device)
        # start of your code
        # This method returns actions the agent output during the training process 
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()   
            noise_scale = max(0.4* (1 - current_episode / max_episode)**0.5, 0.2)  # 从0.3衰减到0.1
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)    
        # end of your code   
        return action
    
    def update(self, state, next_state, action, reward, done, current_episode, max_episode):
        self.buffer.push(state, next_state, action, reward, done)
        if len(self.buffer) < self.batch_size:
            return None, None
        self.total_it += 1
        state, next_state, action, reward, done = self.buffer.sample(self.batch_size, self.device)
        reward = (reward - reward.mean()) / (reward.std() + 1e-6)  #reward normalization

        # start of your code
        # You need to compute actor_loss/critic_loss and implement the backpropagation in this block
        # function you may need: self.actor.zero_grad(), self.actor_opt.step(), actor_loss.backward()
        
        # Add noise to the action
        policy_noise_scale = max(self.policy_noise * (1 - current_episode / max_episode)**0.5, 0.2) #noise decay
        policy_noise = np.random.normal(0, policy_noise_scale) 
        noise = torch.randn_like(action) * torch.tensor(policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
        
        # Compute target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)  
        done = done.float()
        target_Q = reward + (1-done) * self.gamma * target_Q.detach()
        target_Q = target_Q.clamp(min=-10.0, max=10.0)  #Clip the target Q value
        
        # Compute current Q value
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)   
                       
        # Update Critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Delayed policy updates
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            q1, q2 = self.critic(state, self.actor(state))
            actor_loss = -(q1 + q2).mean()  

            # Adjust learning rate 
            if actor_loss.abs().item() > 5.0:  
                self.actor_lr *= 0.8
            elif actor_loss.abs().item() < 1.0:
                self.actor_lr *= 1.2
        
            # Update Actor
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

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

