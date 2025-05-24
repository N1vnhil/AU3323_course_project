from collections import deque
from Agent import PPOAgent
from env import DroneEnv
import numpy as np
from tensorboardX import SummaryWriter
import torch
import logging
from datetime import datetime
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置目录和日志
base_dir = f'results/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(base_dir, exist_ok=True)
for dir_name in ['logs', 'models', 'tensorboard']:
    os.makedirs(f'{base_dir}/{dir_name}', exist_ok=True)

def setup_logging(base_dir):
    """设置同时输出到文件和控制台的日志记录"""
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # 设置文件处理器
    file_handler = logging.FileHandler(f'{base_dir}/logs/training.log')
    file_handler.setFormatter(formatter)

    # 设置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 配置根日志记录器
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging(base_dir)
writer = SummaryWriter(f'{base_dir}/tensorboard')

def format_number(n):
    """格式化数字，添加千位分隔符"""
    return f"{n:,}"


def format_time(seconds):
    """将秒数转换为人类可读的时间格式"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"


def process_batch(trajectory_batch):
    """处理轨迹数据"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转换数据
    states = torch.FloatTensor(np.array([t['state'] for t in trajectory_batch])).to(device)
    actions = torch.FloatTensor(np.array([t['action'] for t in trajectory_batch])).to(device)
    rewards = torch.FloatTensor(np.array([t['reward'] for t in trajectory_batch]).reshape(-1, 1)).to(device)
    next_states = torch.FloatTensor(np.array([t['next_state'] for t in trajectory_batch])).to(device)
    dones = torch.FloatTensor(np.array([t['done'] for t in trajectory_batch]).reshape(-1, 1)).to(device)
    log_probs = torch.FloatTensor(np.array([t['log_prob'] for t in trajectory_batch])).to(device)

    # 添加reward统计信息
    logger.debug(f"Batch reward stats - Mean: {rewards.mean().item():.2f}, "
                f"Min: {rewards.min().item():.2f}, Max: {rewards.max().item():.2f}")

    return states, actions, rewards, next_states, dones, log_probs


def evaluate_policy(env, agent, n_episodes=5):
    """评估策略"""
    try:
        with torch.no_grad():
            agent.eval()
            eval_rewards = []
            eval_lengths = []

            for ep in range(n_episodes):
                state = env.reset()
                total_reward = 0
                steps = 0
                done = False

                # 使用与训练相同的最大步数
                while not done and steps < 700:  # 与训练时相同的步数限制
                    normalized_state = agent.state_normalizer(state)
                    action, _ = agent.get_action(normalized_state)

                    # 记录action用于调试
                    if steps == 0:
                        logger.debug(f"Episode {ep} first action: {action}")

                    next_state, reward, done, info = env.step(action)

                    # 记录每一步的reward用于调试
                    if steps == 0:
                        logger.debug(f"Episode {ep} first step reward: {reward}")

                    total_reward += reward
                    state = next_state
                    steps += 1

                eval_rewards.append(total_reward)
                eval_lengths.append(steps)

            agent.train()

            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            mean_length = np.mean(eval_lengths)

            # 详细的评估信息
            logger.info(f"\nEvaluation Details:")
            logger.info(f"Individual episode rewards: {eval_rewards}")
            logger.info(f"Individual episode lengths: {eval_lengths}")
            logger.info(f"Reward range: [{min(eval_rewards)}, {max(eval_rewards)}]")
            logger.info(f"Average episode length: {mean_length}")

            return mean_reward, std_reward

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        agent.train()
        return -np.inf, 0

def plot_training_curves(stats, base_dir):
    plt.figure(figsize=(15, 5))
    
    # 奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(stats['eval_history'], label='Evaluation Reward')
    plt.plot(stats['avg_rewards'], label='Training Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(stats['actor_losses'], label='Actor Loss')
    plt.plot(stats['critic_losses'], label='Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(f'{base_dir}/training_curves.png')
    plt.close()

def train():
    # 设置随机种子和设备
    torch.manual_seed(53510713690200)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training started on device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 初始化环境和智能体
    env = DroneEnv()
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    agent.actor.to(device)
    agent.critic.to(device)

    # 记录配置信息
    config = {
        'max_episodes': 20000,
        'steps_per_episode': 700,
        'eval_freq': 100,
        'save_freq': 1000
    }

    logger.info("\nTraining Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("------------------------")

    # 训练统计
    stats = {
        'best_reward': -np.inf,
        'best_eval_reward': -np.inf,
        'avg_rewards': deque(maxlen=100),
        'total_steps': 0,
        'episode_lengths': deque(maxlen=100),
        'critic_losses': deque(maxlen=100),
        'actor_losses': deque(maxlen=100),
        'eval_history': [],
        'start_time': datetime.now()
    }

    no_improvement_count = 0
    no_improvement_threshold = 10  # 连续10次评估没有提升就停止

    try:
        for episode in range(config['max_episodes']):
            trajectory_batch = []
            state = env.reset()
            episode_reward = 0
            episode_length = 0

            # 收集轨迹
            for step in range(config['steps_per_episode']):
                action, log_prob = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)

                trajectory_batch.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                    'log_prob': log_prob
                })

                episode_reward += reward
                episode_length += 1
                stats['total_steps'] += 1
                state = next_state

                if done:
                    break

            # 更新策略
            states, actions, rewards, next_states, dones, log_probs = process_batch(trajectory_batch)
            critic_loss, actor_loss = agent.update(states, actions, rewards, next_states, dones, log_probs)

            # 更新统计信息
            stats['avg_rewards'].append(episode_reward)
            stats['episode_lengths'].append(episode_length)
            stats['critic_losses'].append(critic_loss)
            stats['actor_losses'].append(actor_loss)

            # 计算平均值
            avg_reward = np.mean(list(stats['avg_rewards']))
            avg_length = np.mean(list(stats['episode_lengths']))
            avg_critic_loss = np.mean(list(stats['critic_losses']))
            avg_actor_loss = np.mean(list(stats['actor_losses']))

            # 记录训练指标
            writer.add_scalar('training/episode_reward', episode_reward, episode)
            writer.add_scalar('training/average_reward', avg_reward, episode)
            writer.add_scalar('training/episode_length', episode_length, episode)
            writer.add_scalar('loss/critic', critic_loss, episode)
            writer.add_scalar('loss/actor', actor_loss, episode)

            # 简单的进度显示
            if episode % 100 == 0:
                elapsed_time = datetime.now() - stats['start_time']
                elapsed_seconds = elapsed_time.total_seconds()
                steps_per_second = stats['total_steps'] / elapsed_seconds

                logger.info(
                    f"Episode {format_number(episode + 1)}/{format_number(config['max_episodes'])} | "
                    f"Steps: {format_number(stats['total_steps'])} | "
                    f"Reward: {episode_reward:.1f} | "
                    f"Avg(100): {avg_reward:.1f} | "
                    f"Steps/sec: {steps_per_second:.1f}"
                )

            # 定期评估
            if (episode + 1) % config['eval_freq'] == 0:
                eval_mean, eval_std = evaluate_policy(env, agent)
                stats['eval_history'].append(eval_mean)
                writer.add_scalar('eval/mean_reward', eval_mean, episode)

                elapsed_time = datetime.now() - stats['start_time']

                logger.info("\n" + "=" * 50)
                logger.info(f"Evaluation at Episode {format_number(episode + 1)}")
                logger.info(f"Time Elapsed: {format_time(elapsed_time.total_seconds())}")
                logger.info(f"Total Steps: {format_number(stats['total_steps'])}")
                logger.info(f"Recent Training Stats (last 100 episodes):")
                logger.info(f"  Average Reward: {avg_reward:.2f}")
                logger.info(f"  Average Episode Length: {avg_length:.1f}")
                logger.info(f"  Average Actor Loss: {avg_actor_loss:.4f}")
                logger.info(f"  Average Critic Loss: {avg_critic_loss:.4f}")
                logger.info(f"Evaluation Results:")
                logger.info(f"  Current: {eval_mean:.2f} ± {eval_std:.2f}")
                logger.info(f"  Best: {stats['best_eval_reward']:.2f}")
                logger.info("=" * 50 + "\n")

                if eval_mean > stats['best_eval_reward']:
                    stats['best_eval_reward'] = eval_mean
                    agent.save(f'{base_dir}/models/best_model')
                    logger.info(f"New best model saved with reward: {eval_mean:.2f}")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    logger.info(f"No improvement for {no_improvement_count} evaluations")

            # 保存检查点
            if (episode + 1) % config['save_freq'] == 0:
                agent.save(f'{base_dir}/models/checkpoint_{episode + 1}')

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    finally:
        # 训练结束统计
        total_time = datetime.now() - stats['start_time']
        agent.save(f'{base_dir}/models/final_model')

        logger.info("\n" + "=" * 50)
        logger.info("Training Summary:")
        logger.info(f"Total Time: {format_time(total_time.total_seconds())}")
        logger.info(f"Total Episodes: {format_number(episode + 1)}")
        logger.info(f"Total Steps: {format_number(stats['total_steps'])}")
        logger.info(f"Best Evaluation Reward: {stats['best_eval_reward']:.2f}")
        logger.info(f"Final Average Reward (100 ep): {np.mean(list(stats['avg_rewards'])):.2f}")
        logger.info(f"Final Average Episode Length: {np.mean(list(stats['episode_lengths'])):.1f}")
        logger.info(f"Steps per Second: {stats['total_steps'] / total_time.total_seconds():.1f}")
        logger.info("=" * 50)

        env.close()
        writer.close()

        # 绘制训练曲线
        plot_training_curves(stats, base_dir)


if __name__ == "__main__":
    train()