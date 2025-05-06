import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from PIL import Image
import io
 
# 风随机（在一轮飞行中风动态变化，湍流+突风），终点随机，有障碍物
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
        # 定义块状障碍物，每个障碍物由左下角和右上角坐标表示
        self.obstacles = [
            (np.array([15, 15, 0]), np.array([25, 25, 10])),
            (np.array([-15, -15, 0]), np.array([-10, -10, 10]))
        ]
        flag = True
        # 检查终点是否在障碍物内部
        while flag:
            self.target_pos = np.random.uniform(-50, 50, 3)
            self.target_pos[2] = np.abs(self.target_pos[2])
            count = 0
            for min_pos, max_pos in self.obstacles:
                if all(min_pos <= self.target_pos) and all(self.target_pos <= max_pos):
                    break
                else:
                    count += 1
            if count == len(self.obstacles):   
                flag = False
        self.wind_mean = np.random.uniform(-1, 1, 3)  # 平均风速
        self.wind_std = 1.0  # 风速标准差
        self.Lu = 200.0  # 纵向湍流尺度
        self.Lv = 200.0  # 横向湍流尺度
        self.Lw = 50.0   # 垂直湍流尺度
        self.sigma_u = 1.0  # 纵向湍流强度
        self.sigma_v = 1.0  # 横向湍流强度
        self.sigma_w = 1.0  # 垂直湍流强度
        self.Va = 1.0    # 无人机空速
        self.dt = 0.1  # 时间步长
        self.wind_speed = self.wind_mean + np.random.normal(0, self.wind_std, 3)
        self.gust_prob = 0.1  # 突风发生概率
        self.gust_strength = 2.0  # 突风强度
        self.time_step = 0
        self.max_time_steps = 1000

    def dryden_turbulence(self):
        # 纵向湍流
        phi_u = self.sigma_u**2 * (1 + 2 * self.Lu / (self.Va * self.dt)) / (1 + 4 * (self.Lu / (self.Va * self.dt))**2)
        du = np.random.normal(0, np.sqrt(phi_u))

        # 横向湍流
        phi_v = self.sigma_v**2 * (1 + 2 * self.Lv / (self.Va * self.dt)) / (1 + 4 * (self.Lv / (self.Va * self.dt))**2)
        dv = np.random.normal(0, np.sqrt(phi_v))

        # 垂直湍流
        phi_w = self.sigma_w**2 * (1 + 2 * self.Lw / (self.Va * self.dt)) / (1 + 4 * (self.Lw / (self.Va * self.dt))**2)
        dw = np.random.normal(0, np.sqrt(phi_w))

        return np.array([du, dv, dw])

    def step(self, action):
        # 解析动作
        thrust, pitch, roll, yaw = action
        thrust = thrust * 0.5 + 0.5
        pitch = pitch * np.pi/4
        roll = roll * np.pi/4
        yaw = yaw * np.pi/4

        # 模拟湍流：使用Dryden模型
        turbulence = self.dryden_turbulence()
        self.wind_speed = self.wind_mean + turbulence

        # 模拟突风
        if np.random.rand() < self.gust_prob:
            gust_direction = np.random.uniform(-1, 1, 3)
            gust_direction = gust_direction / np.linalg.norm(gust_direction)
            gust = gust_direction * self.gust_strength
            self.wind_speed += gust
        
        wind_force = self.wind_speed * 0.1  

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
        # 检查是否碰撞障碍物
        for min_pos, max_pos in self.obstacles:
            if all(min_pos <= self.drone_pos) and all(self.drone_pos <= max_pos):
                reward -= 5000
                done = True
                break
        else:
            done = False

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
        flag = True
        # 检查终点是否在障碍物内部
        while flag:
            self.target_pos = np.random.uniform(-50, 50, 3)
            self.target_pos[2] = np.abs(self.target_pos[2])
            count = 0
            for min_pos, max_pos in self.obstacles:
                if all(min_pos <= self.target_pos) and all(self.target_pos <= max_pos):
                    break
                else:
                    count += 1
            if count == len(self.obstacles):   
                flag = False
        self.wind_mean = np.random.uniform(-1, 1, 3)  # 平均风速
        self.wind_std = 1.0  # 风速标准差
        self.wind_speed = self.wind_mean + np.random.normal(0, self.wind_std, 3)
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
        for min_pos, max_pos in self.obstacles:
            x = [min_pos[0], max_pos[0], max_pos[0], min_pos[0], min_pos[0]]
            y = [min_pos[1], min_pos[1], max_pos[1], max_pos[1], min_pos[1]]
            z_min = [min_pos[2]] * 5
            z_max = [max_pos[2]] * 5

            vertices = [
                list(zip(x, y, z_min)),
                list(zip(x, y, z_max)),
                [(x[0], y[0], z_min[0]), (x[0], y[0], z_max[0]), (x[1], y[1], z_max[0]), (x[1], y[1], z_min[0])],
                [(x[1], y[1], z_min[0]), (x[1], y[1], z_max[0]), (x[2], y[2], z_max[0]), (x[2], y[2], z_min[0])],
                [(x[2], y[2], z_min[0]), (x[2], y[2], z_max[0]), (x[3], y[3], z_max[0]), (x[3], y[3], z_min[0])],
                [(x[3], y[3], z_min[0]), (x[3], y[3], z_max[0]), (x[0], y[0], z_max[0]), (x[0], y[0], z_min[0])]
            ]
            poly = Poly3DCollection(vertices, alpha=0.5, facecolors='g')
            ax.add_collection3d(poly)

        ax.legend()

        # 将当前图像保存到内存缓冲区
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)

        # 关闭图形
        plt.close(fig)

        return img
