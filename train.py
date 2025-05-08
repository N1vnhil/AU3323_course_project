from collections import deque
from Agent import PPOAgent
from env import DroneEnv
import numpy as np
from tensorboardX import SummaryWriter
from tqdm.rich import tqdm
import torch
import logging
from datetime import datetime
import os


def process_batch(trajectory_batch, config):
    """处理轨迹批次数据"""
    states = np.array([t['state'] for t in trajectory_batch])
    actions = np.array([t['action'] for t in trajectory_batch])
    rewards = np.array([t['reward'] for t in trajectory_batch]).reshape(-1, 1)
    next_states = np.array([t['next_state'] for t in trajectory_batch])
    dones = np.array([t['done'] for t in trajectory_batch]).reshape(-1, 1)
    log_probs = np.array([t['log_prob'] for t in trajectory_batch])

    # 确保数据类型正确
    states = states.astype(np.float32)
    actions = actions.astype(np.float32)
    rewards = rewards.astype(np.float32)
    next_states = next_states.astype(np.float32)
    dones = dones.astype(np.float32)
    log_probs = log_probs.astype(np.float32)

    return states, actions, rewards, next_states, dones, log_probs


def evaluate_policy(env, agent, n_episodes=5):
    """评估当前策略"""
    agent.eval()  # 设置为评估模式
    eval_rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        eval_rewards.append(episode_reward)

    agent.train()  # 恢复训练模式
    return np.mean(eval_rewards), np.std(eval_rewards)


def train():
    # Setup directories
    base_dir = f'results/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    for dir_name in ['logs', 'models', 'tensorboard', 'checkpoints']:
        os.makedirs(f'{base_dir}/{dir_name}', exist_ok=True)

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f'{base_dir}/logs/training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Seeds for reproducibility
    torch.manual_seed(53510713690200)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("Using CPU")

    # Initialize environment and agent
    env = DroneEnv()
    writer = SummaryWriter(f'{base_dir}/tensorboard')

    # Hyperparameters
    config = {
        'state_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.shape[0],
        'gamma': 0.995,
        'actor_lr': 1e-4,
        'critic_lr': 5e-4,  # 修改
        'epsilon': 0.2,
        'epochs': 10,
        'max_episodes': 20000,
        'steps_per_episode': 700,
        'gae_lambda': 0.98,
        'entropy_coef': 0.01,
        'value_clip': 0.2,
        'max_grad_norm': 1.0,  # 修改
        'batch_size': 512,  # 修改
        'eval_freq': 500,
        'save_freq': 100,
        'checkpoint_freq': 1000
    }

    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    logger.info("------------------------")

    # Initialize agent
    agent = PPOAgent(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        gamma=config['gamma'],
        epsilon=config['epsilon'],
        epochs=config['epochs'],
        gae_lambda=config['gae_lambda'],
        entropy_coef=config['entropy_coef'],
        value_clip=config['value_clip'],
        max_grad_norm=config['max_grad_norm']
    )

    if torch.cuda.device_count() > 1:
        agent.actor = torch.nn.DataParallel(agent.actor)
        agent.critic = torch.nn.DataParallel(agent.critic)
    agent.actor.to(device)
    agent.critic.to(device)

    # Training statistics
    stats = {
        'max_score': -np.inf,
        'episode_scores': [],
        'episode_lengths': [],
        'critic_losses': [],
        'actor_losses': [],
        'avg_rewards': deque(maxlen=100),
        'best_avg_reward': -np.inf,
        'training_steps': 0
    }

    try:
        agent.train()  # 确保在训练模式开始
        for episode in tqdm(range(config['max_episodes'])):
            trajectory_batch = []
            state = env.reset()
            episode_reward = 0
            episode_steps = 0

            # Collect trajectory
            for step in range(config['steps_per_episode']):
                try:
                    # Get action
                    action, log_prob = agent.get_action(state)
                    stats['training_steps'] += 1

                    # Take step in environment
                    next_state, reward, done, _ = env.step(action)

                    # Store transition
                    trajectory_batch.append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done,
                        'log_prob': log_prob
                    })

                    episode_reward += reward
                    episode_steps += 1
                    state = next_state

                    if done:
                        break

                except Exception as e:
                    logger.error(f"Error in step {step}: {str(e)}", exc_info=True)
                    break

            # Process batch and update policy
            if len(trajectory_batch) > 0:
                try:
                    states, actions, rewards, next_states, dones, log_probs = process_batch(trajectory_batch, config)

                    # Update policy
                    critic_loss, actor_loss = agent.update(
                        states, actions, rewards, next_states, dones, log_probs
                    )

                    # Update statistics
                    stats['episode_scores'].append(episode_reward)
                    stats['episode_lengths'].append(episode_steps)
                    stats['critic_losses'].append(critic_loss)
                    stats['actor_losses'].append(actor_loss)
                    stats['avg_rewards'].append(episode_reward)

                    # Logging
                    writer.add_scalar('loss/critic', critic_loss, stats['training_steps'])
                    writer.add_scalar('loss/actor', actor_loss, stats['training_steps'])
                    writer.add_scalar('metrics/score', episode_reward, stats['training_steps'])
                    writer.add_scalar('metrics/episode_length', episode_steps, stats['training_steps'])

                    # Periodic evaluation
                    if (episode + 1) % config['eval_freq'] == 0:
                        eval_mean, eval_std = evaluate_policy(env, agent)
                        writer.add_scalar('eval/mean_reward', eval_mean, stats['training_steps'])
                        writer.add_scalar('eval/reward_std', eval_std, stats['training_steps'])

                        logger.info(
                            f"\nEpisode {episode + 1}\n"
                            f"Training Steps: {stats['training_steps']}\n"
                            f"Average Training Reward (last 100): {np.mean(list(stats['avg_rewards'])):.2f}\n"
                            f"Evaluation Reward: {eval_mean:.2f} ± {eval_std:.2f}\n"
                            f"Episode Length: {episode_steps}\n"
                            f"Actor Loss: {actor_loss:.4f}\n"
                            f"Critic Loss: {critic_loss:.4f}\n"
                            f"Best Average Reward: {stats['best_avg_reward']:.2f}"
                        )

                    # Update best score and save model
                    current_avg_reward = np.mean(list(stats['avg_rewards']))
                    if current_avg_reward > stats['best_avg_reward']:
                        stats['best_avg_reward'] = current_avg_reward
                        agent.save(f'{base_dir}/models/best_model')
                        logger.info(f"New best average reward: {current_avg_reward:.2f}")

                    # Periodic checkpoints
                    if (episode + 1) % config['checkpoint_freq'] == 0:
                        agent.save(f'{base_dir}/checkpoints/checkpoint_{episode + 1}')

                    # Early stopping
                    if current_avg_reward >= 200:
                        logger.info(f"Environment solved in {episode + 1} episodes!")
                        agent.save(f'{base_dir}/models/final_model')
                        break

                except Exception as e:
                    logger.error(f"Error in update step: {str(e)}", exc_info=True)
                    continue

            # Cleanup
            if (episode + 1) % 100 == 0:
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.warning(f"Training interrupted at episode {episode + 1}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
    finally:
        # Save final state
        if len(stats['episode_scores']) > 0:
            final_path = f'{base_dir}/models/final_model'
            agent.save(final_path)

            logger.info("\nTraining Summary:")
            logger.info(f"Total Episodes: {episode + 1}")
            logger.info(f"Total Steps: {stats['training_steps']}")
            logger.info(f"Best Average Reward: {stats['best_avg_reward']:.2f}")
            logger.info(f"Final Average Reward (100 episodes): {np.mean(list(stats['avg_rewards'])):.2f}")
            logger.info(f"Model saved to {final_path}")
        else:
            logger.error("No episodes completed successfully")

        env.close()
        writer.close()


if __name__ == "__main__":
    train()