import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from PIL import Image
import io
 
# 无风，终点固定，无障碍物
class DroneEnv(gym.Env):
    def __init__(self):
        # 定义动作空间：推力、俯仰角、横滚角、偏航角
        self.action_space = spaces.Box(low=np.array([0, -np.pi/4, -np.pi/4, -np.pi/4]),
                                       high=np.array([1, np.pi/4, np.pi/4, np.pi/4]),
                                       dtype=np.float32)
        # 定义观测空间：无人机位置、速度、目标位置、风速
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # 初始化环境参数
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        # 大家可以在这里自由修改目标位置
        self.target_pos = np.array([-15, 30, 20])
        self.wind_speed = np.zeros(3) 
        # 定义块状障碍物，每个障碍物由左下角和右上角坐标表示
        self.obstacles = [
            (np.array([15, 15, 0]), np.array([25, 25, 10])),
            (np.array([-15, -15, 0]), np.array([-10, -10, 10]))
        ]
        self.time_step = 0
        self.max_time_steps = 2000

    def step(self, action):
        # 解析动作
        thrust, pitch, roll, yaw = action
        thrust = thrust * 0.5 + 0.5
        pitch = pitch * np.pi/4
        roll = roll * np.pi/4
        yaw = yaw * np.pi/4

        # 模拟风的影响
        wind_force = self.wind_speed * 0.1
        rotation_matrix = np.array([
            [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
            [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
            [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
        ])
        thrust_vector = np.array([0,0,thrust])
        acceleration = np.dot(rotation_matrix, thrust_vector) + wind_force
        self.drone_vel += acceleration * 0.1
        self.drone_pos += self.drone_vel * 0.1

        # 计算各种距离和速度
        distance_to_target = np.linalg.norm(self.drone_pos - self.target_pos)
        velocity = np.linalg.norm(self.drone_vel)
        
        # 计算到目标的方向向量
        direction_to_target = self.target_pos - self.drone_pos
        direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-8)
        
        # 计算速度在目标方向上的投影
        velocity_projection = np.dot(self.drone_vel, direction_to_target)
        
        # 计算姿态角
        attitude_angles = np.array([pitch, roll, yaw])
        attitude_penalty = np.linalg.norm(attitude_angles) * 0.1

        # 1. 距离奖励：使用负的指数函数，使奖励更平滑
        distance_reward = 10.0 * np.exp(-0.01 * distance_to_target)
        
        # 2. 速度奖励：鼓励向目标方向移动
        velocity_reward = 5.0 * velocity_projection
        
        # 3. 姿态奖励：惩罚过大的姿态角
        attitude_reward = -attitude_penalty
        
        # 4. 高度保持奖励：特别关注高度控制
        height_diff = abs(self.drone_pos[2] - self.target_pos[2])
        height_reward = 5.0 * np.exp(-0.1 * height_diff)
        
        # 5. 稳定性奖励：惩罚过大的速度
        stability_reward = -0.1 * velocity
        
        # 6. 能量效率奖励：惩罚过大的推力
        energy_reward = -0.1 * thrust
        
        # 7. 成功奖励：当接近目标时给予额外奖励
        success_reward = 0
        if distance_to_target < 1:
            success_reward = 1000
            print('Success!')
            done = False
        else:
            done = False

        # 组合所有奖励
        reward = (distance_reward + 
                 velocity_reward + 
                 attitude_reward + 
                 height_reward + 
                 stability_reward + 
                 energy_reward + 
                 success_reward)

        # 检查是否超过最大时间步
        self.time_step += 1
        if self.time_step >= self.max_time_steps:
            done = True

        # 生成观测
        observation = np.concatenate([self.drone_pos, self.drone_vel, self.target_pos, self.wind_speed])

        return observation, reward, done, {}

    def reset(self):
        # 重置环境
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        # 目标位置在这里也需要同步修改
        self.target_pos = np.array([-15, 30, 20])
        self.wind_speed = np.zeros(3)
        self.time_step = 0

        observation = np.concatenate([self.drone_pos, self.drone_vel, self.target_pos, self.wind_speed])
        return observation

    def render(self, mode='human'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制无人机
        ax.scatter(self.drone_pos[0], self.drone_pos[1], self.drone_pos[2], c='b', label='Drone')

        # 绘制目标
        ax.scatter(self.target_pos[0], self.target_pos[1], self.target_pos[2], c='r', label='Target')

        # 绘制障碍物
        # for min_pos, max_pos in self.obstacles:
        #     x = [min_pos[0], max_pos[0], max_pos[0], min_pos[0], min_pos[0]]
        #     y = [min_pos[1], min_pos[1], max_pos[1], max_pos[1], min_pos[1]]
        #     z_min = [min_pos[2]] * 5
        #     z_max = [max_pos[2]] * 5

        #     vertices = [
        #         list(zip(x, y, z_min)),
        #         list(zip(x, y, z_max)),
        #         [(x[0], y[0], z_min[0]), (x[0], y[0], z_max[0]), (x[1], y[1], z_max[0]), (x[1], y[1], z_min[0])],
        #         [(x[1], y[1], z_min[0]), (x[1], y[1], z_max[0]), (x[2], y[2], z_max[0]), (x[2], y[2], z_min[0])],
        #         [(x[2], y[2], z_min[0]), (x[2], y[2], z_max[0]), (x[3], y[3], z_max[0]), (x[3], y[3], z_min[0])],
        #         [(x[3], y[3], z_min[0]), (x[3], y[3], z_max[0]), (x[0], y[0], z_max[0]), (x[0], y[0], z_min[0])]
        #     ]
        #     poly = Poly3DCollection(vertices, alpha=0.5, facecolors='g')
        #     ax.add_collection3d(poly)

        ax.legend()

        # 将当前图像保存到内存缓冲区
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)

        # 关闭图形
        plt.close(fig)

        return img
