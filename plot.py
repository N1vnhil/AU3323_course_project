import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def smooth(data, weight=0.7):
    """指数移动平均平滑函数"""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# 指定 TensorBoard 事件文件路径
event_file = r"D:\zhangyx\大作业\选题一 无人机飞行控制\CourseProject25\runs\env3-4MLP\events.out.tfevents.1748442613.LAPTOP-FGTE2DQ2"
# 加载事件数据
ea = EventAccumulator(event_file)
ea.Reload()

# 检查可用标签
tags = ea.Tags()["scalars"]
print("可用 tags:", tags)

# 提取某个 scalar 的步数和数值
def get_scalar(tag):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)

# 提取并平滑 Actor/Critic Loss 和 Score
actor_steps, actor_vals = get_scalar("loss/actor")
critic_steps, critic_vals = get_scalar("loss/critic")
score_steps, score_vals = get_scalar("score")

actor_vals_smooth = smooth(actor_vals)
critic_vals_smooth = smooth(critic_vals)
score_vals_smooth = smooth(score_vals)

# ===== 图 1: Actor 和 Critic Loss =====
plt.figure(figsize=(9, 5))
plt.plot(actor_steps, actor_vals, color="blue", alpha=0.3, label="Actor Loss (raw)")
plt.plot(actor_steps, actor_vals_smooth, color="blue", label="Actor Loss (smoothed)")

plt.plot(critic_steps, critic_vals, color="red", alpha=0.3, label="Critic Loss (raw)")
plt.plot(critic_steps, critic_vals_smooth, color="red", label="Critic Loss (smoothed)")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Actor & Critic Loss Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== 图 2: Score =====
plt.figure(figsize=(9, 5))
plt.plot(score_steps, score_vals, color="green", alpha=0.3, label="Score (raw)")
plt.plot(score_steps, score_vals_smooth, color="green", label="Score (smoothed)")

plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Score Over Episodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
