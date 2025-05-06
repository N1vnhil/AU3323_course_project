from Agent import DDPGAgent
from env import DroneEnv  # 导入新环境
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
BATCH_SIZE = 128
GAMMA = 0.99 
ACTOR_LR = 1e-4 
CRITIC_LR = 1e-3 
TAU = 1e-3
MAX_EPISODE = 3000
T = 700

agent = DDPGAgent(STATE_DIM, ACTION_DIM, ACTOR_LR, CRITIC_LR, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU)
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
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            l1, l2 = agent.update(state, next_state, action, reward, done)      
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
        if score > 200:
            break
except KeyboardInterrupt:
    print(f"Keyboard Terminated. Trained for {i + 1} episodes.")
finally:
    env.close()