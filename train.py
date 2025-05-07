from Agent import DDPGAgent, PPOAgent
from env import DroneEnv
import numpy as np
from tensorboardX import SummaryWriter
from tqdm.rich import tqdm
import torch
import logging
from datetime import datetime
import os

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging configuration
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_{current_time}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up device configuration
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda")
    logger.info(f"Using {num_gpus} GPUs")
    for i in range(num_gpus):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    device = torch.device("cpu")
    logger.info("Using CPU")

torch.manual_seed(53510713690200)
writer = SummaryWriter(f'runs/training_{current_time}')
env = DroneEnv()
env.reset()
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# PPO Hyperparameters
# PPO Hyperparameters
GAMMA = 0.99
ACTOR_LR = 1e-4  # Reduced from 3e-4
CRITIC_LR = 3e-4  # Reduced from 1e-3
EPSILON = 0.1  # Reduced from 0.2
EPOCHS = 5  # Reduced from 10
MAX_EPISODE = 20000
T = 700
LOG_INTERVAL = 100

# Log hyperparameters
logger.info("Training Configuration:")
logger.info(f"State Dimension: {STATE_DIM}")
logger.info(f"Action Dimension: {ACTION_DIM}")
logger.info(f"Gamma: {GAMMA}")
logger.info(f"Actor Learning Rate: {ACTOR_LR}")
logger.info(f"Critic Learning Rate: {CRITIC_LR}")
logger.info(f"PPO Epsilon: {EPSILON}")
logger.info(f"PPO Epochs: {EPOCHS}")
logger.info(f"Max Episodes: {MAX_EPISODE}")
logger.info(f"Max Steps per Episode: {T}")
logger.info("------------------------")

# Initialize agent and move to device
agent = PPOAgent(STATE_DIM, ACTION_DIM, ACTOR_LR, CRITIC_LR, GAMMA, EPSILON, EPOCHS)
# agent.load("./checkpoints/ppo_actor_2025-05-07 08:17:17.098576.pth",
#            "./checkpoints/ppo_critic_2025-05-07 08:17:17.099961.pth")

if torch.cuda.device_count() > 1:
    agent.actor = torch.nn.DataParallel(agent.actor)
    agent.critic = torch.nn.DataParallel(agent.critic)
agent.actor.to(device)
agent.critic.to(device)

max_score = -np.inf

# Initialize statistics tracking
stats_window_size = 100
episode_scores = []
episode_lengths = []
critic_losses = []
actor_losses = []

try:
    for i in tqdm(range(MAX_EPISODE)):
        score = 0
        state = env.reset()
        episode_steps = 0

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []

        # Collect trajectory
        for j in range(T):
            try:
                action, log_prob = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_steps += 1

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                log_probs.append(log_prob)

                score += reward
                state = next_state

                if done:
                    break

            except Exception as e:
                logger.error(f"Error in step {j} of episode {i}: {str(e)}", exc_info=True)
                break

        if len(states) == 0:
            logger.warning(f"Episode {i} collected no transitions, skipping update")
            continue

        # Convert lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)
        log_probs = np.array(log_probs)

        # Update policy
        try:
            critic_loss, actor_loss = agent.update(
                states, actions, rewards, next_states, dones, log_probs
            )

            # Store statistics
            episode_scores.append(score)
            episode_lengths.append(episode_steps)
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

            # Log to tensorboard
            writer.add_scalar('loss/critic', critic_loss, i)
            writer.add_scalar('loss/actor', actor_loss, i)
            writer.add_scalar('metrics/score', score, i)
            writer.add_scalar('metrics/episode_length', episode_steps, i)

        except Exception as e:
            logger.error(f"Error in update step of episode {i}: {str(e)}", exc_info=True)
            continue

        # Log to console and file every LOG_INTERVAL episodes
        if (i + 1) % LOG_INTERVAL == 0:
            recent_scores = episode_scores[-min(stats_window_size, len(episode_scores)):]
            recent_lengths = episode_lengths[-min(stats_window_size, len(episode_lengths)):]
            recent_critic_losses = critic_losses[-min(stats_window_size, len(critic_losses)):]
            recent_actor_losses = actor_losses[-min(stats_window_size, len(actor_losses)):]

            avg_score = np.mean(recent_scores)
            avg_length = np.mean(recent_lengths)
            avg_critic_loss = np.mean(recent_critic_losses)
            avg_actor_loss = np.mean(recent_actor_losses)

            logger.info(f"\nEpisode {i + 1}/{MAX_EPISODE}")
            logger.info(f"Last {len(recent_scores)} episodes statistics:")
            logger.info(f"Average Score: {avg_score:.2f}")
            logger.info(f"Average Episode Length: {avg_length:.2f}")
            logger.info(f"Average Critic Loss: {avg_critic_loss:.4f}")
            logger.info(f"Average Actor Loss: {avg_actor_loss:.4f}")
            logger.info(f"Best Score So Far: {max_score:.2f}")
            logger.info("------------------------")

        # Save best model
        if score > max_score:
            agent.save()
            max_score = score
            logger.info(f"New best score: {max_score:.2f}")

        # Early stopping
        if score > 200:
            logger.info(f"Environment solved in {i + 1} episodes!")
            break

except KeyboardInterrupt:
    logger.warning(f"Training interrupted at episode {i + 1}")
except Exception as e:
    logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
finally:
    # Log final statistics
    if len(episode_scores) > 0:
        logger.info("\nTraining Summary:")
        logger.info(f"Total Episodes: {i + 1}")
        logger.info(f"Best Score: {max_score:.2f}")
        recent_scores = episode_scores[-min(stats_window_size, len(episode_scores)):]
        recent_lengths = episode_lengths[-min(stats_window_size, len(episode_lengths)):]
        logger.info(f"Final Average Score ({len(recent_scores)} episodes): {np.mean(recent_scores):.2f}")
        logger.info(f"Final Average Episode Length: {np.mean(recent_lengths):.2f}")
    else:
        logger.error("No episodes completed successfully")

    env.close()
    writer.close()
