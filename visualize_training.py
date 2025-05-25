import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file):
    eval_episodes = []  # evaluation episode numbers
    eval_rewards = []   # average reward at evaluation
    actor_losses = []   # actor loss at evaluation
    critic_losses = []  # critic loss at evaluation
    
    # 更健壮的正则表达式，支持整数、浮点数和科学计数法
    number_pattern = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Parse evaluation episode number
            if 'Evaluation at Episode' in line:
                match = re.search(r'Evaluation at Episode ([\d,]+)', line)
                if match:
                    eval_episode = int(match.group(1).replace(',', ''))
                    eval_episodes.append(eval_episode)
                    
                    # Initialize temporary variables for this evaluation
                    reward = None
                    actor_loss = None
                    critic_loss = None
                    
                    # Look for Average Reward, Actor Loss, and Critic Loss in the next 15 lines
                    for j in range(i, min(i + 15, len(lines))):
                        if 'Average Reward:' in lines[j]:
                            match = re.search(rf'Average Reward: {number_pattern}', lines[j])
                            if match:
                                reward = float(match.group(1))
                        if 'Average Actor Loss:' in lines[j]:
                            match = re.search(rf'Average Actor Loss: {number_pattern}', lines[j])
                            if match:
                                actor_loss = float(match.group(1))
                        if 'Average Critic Loss:' in lines[j]:
                            match = re.search(rf'Average Critic Loss: {number_pattern}', lines[j])
                            if match:
                                critic_loss = float(match.group(1))
                    
                    # Only append if all values are found
                    if reward is not None and actor_loss is not None and critic_loss is not None:
                        eval_rewards.append(reward)
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)
                    else:
                        print(f"Warning: Incomplete data at episode {eval_episode}. Skipping.")
                        eval_episodes.pop()  # Remove the episode if data is incomplete
            
            i += 1
    
    # Verify data integrity
    if not eval_episodes:
        raise ValueError("No evaluation data found in log file.")
    
    # Check if episodes are in ascending order
    if not all(eval_episodes[i] <= eval_episodes[i+1] for i in range(len(eval_episodes)-1)):
        print("Warning: Episode numbers are not in ascending order.")
    
    # Check for NaN or Inf
    for arr, name in [(eval_rewards, "rewards"), (actor_losses, "actor losses"), (critic_losses, "critic losses")]:
        if any(np.isnan(x) or np.isinf(x) for x in arr):
            print(f"Warning: {name} contains NaN or Inf values.")
    
    return eval_episodes, eval_rewards, actor_losses, critic_losses

def plot_training_curves(eval_episodes, eval_rewards, actor_losses, critic_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot evaluation rewards
    ax1.plot(eval_episodes, eval_rewards, label='Evaluation Reward', linewidth=2, markersize=8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Evaluation Reward During Training')
    ax1.grid(True)
    ax1.legend()
    
    # Set y-axis limits for rewards if they are negative
    if eval_rewards:
        reward_min = min(eval_rewards)
        reward_max = max(eval_rewards)
        if reward_min < 0:
            ax1.set_ylim(reward_min * 1.001, reward_max * 0.999 if reward_max < 0 else reward_max * 1.05)
    
    # Plot losses with twin y-axes for different scales
    ax2.plot(eval_episodes, actor_losses, 'r-', label='Actor Loss', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Actor Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.grid(True)
    
    # Create a second y-axis for critic loss
    ax2_twin = ax2.twinx()
    ax2_twin.plot(eval_episodes, critic_losses, 'g-', label='Critic Loss', linewidth=2)
    ax2_twin.set_ylabel('Critic Loss', color='g')
    ax2_twin.tick_params(axis='y', labelcolor='g')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax2.set_title('Training Loss During Evaluation')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    log_file = 'results/20250525_141226/logs/training.log'
    try:
        eval_episodes, eval_rewards, actor_losses, critic_losses = parse_log_file(log_file)
        print(f"Found {len(eval_episodes)} evaluation points")
        print(f"Episodes: {eval_episodes}")
        print(f"Rewards: {eval_rewards}")
        print(f"Actor Losses: {actor_losses}")
        print(f"Critic Losses: {critic_losses}")
        plot_training_curves(eval_episodes, eval_rewards, actor_losses, critic_losses)
        print("Training curves saved as 'training_curves.png'")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()