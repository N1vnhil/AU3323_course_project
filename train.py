from Agent import TD3Agent
from env_list.env3 import DroneEnv  # 导入新环境
import numpy as np
from tensorboardX import SummaryWriter
from tqdm.rich import tqdm
import torch

torch.manual_seed(53510713690200)
writer = SummaryWriter()
env = DroneEnv()  # 使用新环境
env.reset()
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
BUFFER_SIZE = 1000000
BATCH_SIZE = 512
GAMMA = 0.99 
ACTOR_LR = 5e-5 
CRITIC_LR = 1e-5 
TAU = 5e-3
POLICY_NOISE = 0.4
NOISE_CLIP = 0.25
POLICY_FREQ = 3
MAX_EPISODE = 800
T = 400

agent = TD3Agent(STATE_DIM, ACTION_DIM, ACTOR_LR, CRITIC_LR, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, POLICY_NOISE, NOISE_CLIP, POLICY_FREQ)
# 第一次训练时，先注释掉下面这一行，想用之前训练的参数继续训练可以取消注释
# agent.load('checkpoints/actor.pth', 'checkpoints/critic.pth', 'checkpoints/actor_target.pth', 'checkpoints/critic_target.pth')
count = 0
max_score = -np.inf
try:
    for i in tqdm(range(MAX_EPISODE)):
        score = 0
        state = env.reset()
        l1s, l2s = [], []
        for j in range(T):
            action_tensor = agent.get_action(state, i, MAX_EPISODE)
            
            if isinstance(action_tensor, torch.Tensor):
                action = action_tensor.detach().cpu().numpy().squeeze()
            else:
                action = np.asarray(action_tensor).squeeze()
            
            assert action.shape == (ACTION_DIM,), f"action shape expected {(ACTION_DIM,)}, got {action.shape}"

            next_state, reward, done, _ = env.step(action)
            l1, l2 = agent.update(state, next_state, action, reward, done, i, MAX_EPISODE)  
            score += reward
            state = next_state
            if l1 is not None and l2 is not None:
                l1s.append(l1.item())
                l2s.append(l2.item())
                writer.add_scalar('loss/critic', l1, count)
                writer.add_scalar('loss/actor', l2, count)                             
                count += 1
            if done:
                break
        writer.add_scalar('score', score, i + 1)
        if score > max_score:
            agent.save()
            max_score = score

except KeyboardInterrupt:
    print(f"Keyboard Terminated. Trained for {i + 1} episodes.")
finally:
    env.close()

