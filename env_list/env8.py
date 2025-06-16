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
        self.action_space = spaces.Box(low=np.array([0, -np.pi / 2, -np.pi / 2, -np.pi / 2]),
                                       high=np.array([1, np.pi / 2, np.pi / 2, np.pi / 2]),
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
        self.Lw = 50.0  # 垂直湍流尺度
        self.sigma_u = 1.0  # 纵向湍流强度
        self.sigma_v = 1.0  # 横向湍流强度
        self.sigma_w = 1.0  # 垂直湍流强度
        self.Va = 1.0  # 无人机空速
        self.dt = 0.1  # 时间步长
        self.wind_speed = self.wind_mean + np.random.normal(0, self.wind_std, 3)
        self.gust_prob = 0.1  # 突风发生概率
        self.gust_strength = 2.0  # 突风强度
        self.time_step = 0
        self.max_time_steps = 1000

    def dryden_turbulence(self):
        # 纵向湍流
        phi_u = self.sigma_u ** 2 * (1 + 2 * self.Lu / (self.Va * self.dt)) / (
                    1 + 4 * (self.Lu / (self.Va * self.dt)) ** 2)
        du = np.random.normal(0, np.sqrt(phi_u))

        # 横向湍流
        phi_v = self.sigma_v ** 2 * (1 + 2 * self.Lv / (self.Va * self.dt)) / (
                    1 + 4 * (self.Lv / (self.Va * self.dt)) ** 2)
        dv = np.random.normal(0, np.sqrt(phi_v))

        # 垂直湍流
        phi_w = self.sigma_w ** 2 * (1 + 2 * self.Lw / (self.Va * self.dt)) / (
                    1 + 4 * (self.Lw / (self.Va * self.dt)) ** 2)
        dw = np.random.normal(0, np.sqrt(phi_w))

        return np.array([du, dv, dw])

    def step(self, action):
        thrust, pitch, roll, yaw = action

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
        rotation_matrix = np.array([
            [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
             np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
            [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
             np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
            [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
        ])
        thrust_vector = np.array([0, 0, thrust])
        gravity = np.array([0, 0, -0.3])
        acceleration = np.dot(rotation_matrix, thrust_vector) + wind_force + gravity
        self.drone_vel += acceleration * self.dt
        self.drone_pos += self.drone_vel * self.dt

        self.drone_pos = np.clip(self.drone_pos, -50, 50)
        self.drone_vel = np.clip(self.drone_vel, -20, 20)

        # 计算距离奖励
        distance_to_target = np.linalg.norm(self.drone_pos - self.target_pos)
        max_distance = 100.0  # 环境的最大可能距离
        distance_reward = -distance_to_target / max_distance  # 归一化到[-1, 0]

        # 计算避障奖励
        avoidance_reward = 0
        min_distance_to_obstacle = float('inf')
        safe_distance = 5.0  # 安全距离阈值
        for min_pos, max_pos in self.obstacles:
            closest_point = np.maximum(min_pos, np.minimum(self.drone_pos, max_pos))
            distance = np.linalg.norm(self.drone_pos - closest_point)
            min_distance_to_obstacle = min(min_distance_to_obstacle, distance)
        avoidance_reward = -np.exp(-min_distance_to_obstacle / safe_distance)  # 归一化到[-1, 0]

        # 计算速度方向奖励
        direction_to_target = self.target_pos - self.drone_pos
        direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-8)
        velocity = np.linalg.norm(self.drone_vel)
        velocity_projection = np.dot(self.drone_vel, direction_to_target)
        velocity_direction_reward = velocity_projection / (velocity + 1e-8) - 0.05 * velocity ** 2  # 速度惩罚

        # 计算风力适应奖励
        wind_adaptation_reward = -np.dot(self.drone_vel, self.wind_speed) / (velocity + 1e-8)

        # 成功奖励
        success_reward = 0
        done = False
        if distance_to_target < 1:
            success_reward = 1000
            print('Success!')
            done = True
        elif distance_to_target <= 2:
            success_reward = 500
        elif distance_to_target <= 3:
            success_reward = 200
        elif distance_to_target <= 5:
            success_reward = 100

        # 合并奖励
        w1, w3, w4, w5 = 1, 1, 0.4, 1  # 权重
        reward = (w1 * distance_reward + w3 * avoidance_reward +
                  w4 * velocity_direction_reward + w5 * wind_adaptation_reward + success_reward)

        # 记录奖励构成
        # if self.time_step % 100 == 0:  # 每100步记录一次
        #     print(f"\n奖励函数构成 (Step {self.time_step}):")
        #     print(f"距离奖励: {distance_reward:.2f}")
        #     print(f"速度奖励: {velocity_direction_reward:.2f}")
        #     print(f"避障奖励：{avoidance_reward:.2f}")
        #     print(f"风力适应奖励：{wind_adaptation_reward:.2f}")
        #     print(f"成功奖励: {success_reward:.2f}")
        #     print(f"总奖励: {reward:.2f}")
        #     print(f"当前距离: {distance_to_target:.2f}")
        #     print("="*50)


        # 检查是否碰撞障碍物
        for min_pos, max_pos in self.obstacles:
            if all(min_pos <= self.drone_pos) and all(self.drone_pos <= max_pos):
                reward -= 10  # 碰撞惩罚
                done = True
                break

        # 检查是否超过最大时间步
        self.time_step += 1
        if self.time_step >= self.max_time_steps:
            done = True
            reward -= 5  # 超时惩罚

        # 保存状态用于下次比较
        self.prev_distance = distance_to_target
        self.prev_vel = self.drone_vel.copy()

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