from Agent import PPOAgent
from env import DroneEnv
from matplotlib import pyplot as plt
from matplotlib import animation
from datetime import datetime
import numpy as np
import os

# 创建保存动画的目录
os.makedirs('./animations', exist_ok=True)

env = DroneEnv()
env.reset()
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# 初始化智能体
agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

# 加载模型
checkpoint_path = 'results/20250525_013718/models/best_model'
agent.load(checkpoint_path)

trajectory = []
ret = 0

state = env.reset()
for j in range(env.max_time_steps): 
    action, _, _ = agent.get_action(state) 
    next_state, reward, done, _ = env.step(action)
    drone_pos = next_state[:3]
    trajectory.append(drone_pos)
    state = next_state
    ret += reward
    if done:
        break

env.close()

# 创建图形
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

# 设置坐标轴范围
max_range_xy = max(abs(env.target_pos[0:2].max()), abs(env.target_pos[0:2].min())) * 1.5
max_range_z = max(abs(env.target_pos[2]), 0) * 20  # 增加z轴的范围

ax.set_xlim([-max_range_xy, max_range_xy])
ax.set_ylim([-max_range_xy, max_range_xy])
ax.set_zlim([0, max_range_z])  # 调整z轴范围

# 设置标签
ax.set_xlabel('X', fontsize=12, labelpad=10)
ax.set_ylabel('Y', fontsize=12, labelpad=10)
ax.set_zlabel('Z', fontsize=12, labelpad=10)

# 设置刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 添加网格
ax.grid(True, alpha=0.3)

trajectory = np.array(trajectory)

def update(frame):
    ax.cla()

    # 重新设置视角和范围
    ax.set_xlim([-max_range_xy, max_range_xy])
    ax.set_ylim([-max_range_xy, max_range_xy])
    ax.set_zlim([0, max_range_z])

    # 设置标签和网格
    ax.set_xlabel('X', fontsize=12, labelpad=10)
    ax.set_ylabel('Y', fontsize=12, labelpad=10)
    ax.set_zlabel('Z', fontsize=12, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3)

    # 添加参考平面
    x_range = np.array([-max_range_xy, max_range_xy])
    y_range = np.array([-max_range_xy, max_range_xy])
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')

    # 绘制目标点
    ax.scatter(env.target_pos[0], env.target_pos[1], env.target_pos[2],
               c='red', marker='*', s=200, label='Target')

    # 绘制轨迹
    if frame > 0:
        ax.plot(trajectory[:frame, 0], trajectory[:frame, 1], trajectory[:frame, 2],
                c='blue', alpha=0.6, linewidth=2, label='Trajectory')

    # 绘制当前位置
    ax.scatter(trajectory[frame, 0], trajectory[frame, 1], trajectory[frame, 2],
               c='green', s=150, label='Drone')

    # 添加图例
    ax.legend(fontsize=12)

    # 设置视角
    ax.view_init(elev=25, azim=45)  # 调整仰角，使z轴更容易观察

    # 设置坐标轴刻度间隔
    ax.set_xticks(np.linspace(-max_range_xy, max_range_xy, 9))
    ax.set_yticks(np.linspace(-max_range_xy, max_range_xy, 9))
    ax.set_zticks(np.linspace(0, max_range_z, 11))  # 增加z轴刻度数量

    return ax,

ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                              interval=50, blit=False)

ani.save(f'./animations/drone_animation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4',
         writer='ffmpeg', fps=20)