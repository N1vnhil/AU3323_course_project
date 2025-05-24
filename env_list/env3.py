import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from PIL import Image
import io
 
# 风固定，终点随机，无障碍物
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
        self.target_pos = np.random.uniform(-50, 50, 3)
        self.target_pos[2] = np.abs(self.target_pos[2])
        # 大家可以在这里修改风速
        self.wind_speed = np.array([0.5, -0.3, -0.1])
        # 定义块状障碍物，每个障碍物由左下角和右上角坐标表示
        self.obstacles = [
            (np.array([15, 15, 0]), np.array([25, 25, 10])),
            (np.array([-15, -15, 0]), np.array([-10, -10, 10]))
        ]
        self.time_step = 0
        self.max_time_steps = 1000

    def step(self, action):
        # 解析动作
        thrust, pitch, roll, yaw = action
        thrust = thrust * 0.5 + 0.5
        pitch = pitch * np.pi/4
        roll = roll * np.pi/4
        yaw = yaw * np.pi/4

        # 模拟风的影响
        wind_force = self.wind_speed * 0.1  # 简单模拟风的作用力

        # 更新无人机状态
        # 动力学模型
        rotation_matrix = np.array([
            [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
            [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
            [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
        ])
        thrust_vector = np.array([0,0,thrust])
        acceleration = np.dot(rotation_matrix, thrust_vector) + wind_force
        self.drone_vel += acceleration * 0.1
        self.drone_pos += self.drone_vel * 0.1
        

        # 计算奖励
        distance_to_target = np.linalg.norm(self.drone_pos - self.target_pos)
        reward = 1 - 0.4*distance_to_target

        done = False
        # # 检查是否碰撞障碍物
        # for min_pos, max_pos in self.obstacles:
        #     if all(min_pos <= self.drone_pos) and all(self.drone_pos <= max_pos):
        #         reward -= 2000
        #         done = True
        #         break
        # else:
        #     done = False

        # 检查是否到达目标
        if distance_to_target < 1:
            reward += 10000
            print('Success!')
            done = True

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
        self.target_pos = np.random.uniform(-50, 50, 3)
        self.target_pos[2] = np.abs(self.target_pos[2])
        # 风速在这里需要同步修改
        self.wind_speed = np.array([0.5, -0.3, -0.1])
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
