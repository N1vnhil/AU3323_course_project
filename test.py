from Agent import DDPGAgent
from env import DroneEnv
from matplotlib import pyplot as plt
from matplotlib import animation

env = DroneEnv()
env.reset()
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
GAMMA = 0.99
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
TAU = 1e-4
MAX_EPISODE = 3000
T = 7000

agent = DDPGAgent(STATE_DIM, ACTION_DIM, ACTOR_LR, CRITIC_LR, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU)
agent.load('checkpoints/actor.pth', 'checkpoints/critic.pth', 'checkpoints/actor_target.pth', 'checkpoints/critic_target.pth')

ret = 0
frames = []

state = env.reset()
for j in range(T):
    frame = env.render()
    action = agent.get_action(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    ret += reward
    frames.append(frame)
    if done:
        break

env.close()

# 创建动画
fig = plt.figure()
plt.axis('off')
ims = [[plt.imshow(frame, animated=True)] for frame in frames]
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

plt.show()


    
