from Agent import PPOAgent
from real_env import DroneEnv
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

# Create directory for saving animations
os.makedirs('./animations', exist_ok=True)

env = DroneEnv()
env.reset()
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# Initialize agent
agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

# Load model
checkpoint_path = 'results/20250526_081054/models/best_model'
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

# Create figure
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

# Set axis ranges based on environment boundaries
max_range_xy = 50 * 1.5  # Based on env's position clipping (-50, 50)
max_range_z = max(abs(env.target_pos[2]), 10) * 1.5  # Adjust z-axis based on target height

ax.set_xlim([-max_range_xy, max_range_xy])
ax.set_ylim([-max_range_xy, max_range_xy])
ax.set_zlim([0, max_range_z])

# Set labels
ax.set_xlabel('X', fontsize=12, labelpad=10)
ax.set_ylabel('Y', fontsize=12, labelpad=10)
ax.set_zlabel('Z', fontsize=12, labelpad=10)

# Set tick font size
ax.tick_params(axis='both', which='major', labelsize=10)

# Add grid
ax.grid(True, alpha=0.3)

trajectory = np.array(trajectory)

def update(frame):
    ax.cla()

    # Reset axis ranges
    ax.set_xlim([-max_range_xy, max_range_xy])
    ax.set_ylim([-max_range_xy, max_range_xy])
    ax.set_zlim([0, max_range_z])

    # Set labels and grid
    ax.set_xlabel('X', fontsize=12, labelpad=10)
    ax.set_ylabel('Y', fontsize=12, labelpad=10)
    ax.set_zlabel('Z', fontsize=12, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3)

    # Add reference plane at z=0
    x_range = np.array([-max_range_xy, max_range_xy])
    y_range = np.array([-max_range_xy, max_range_xy])
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')

    # Draw obstacles (matching env.py's render method)
    for min_pos, max_pos in env.obstacles:
        x = [min_pos[0], max_pos[0], max_pos[0], min_pos[0], min_pos[0]]
        y = [min_pos[1], min_pos[1], max_pos[1], max_pos[1], min_pos[1]]
        z_min = [min_pos[2]] * 5
        z_max = [max_pos[2]] * 5

        vertices = [
            list(zip(x, y, z_min)),  # Bottom face
            list(zip(x, y, z_max)),  # Top face
            [(x[0], y[0], z_min[0]), (x[0], y[0], z_max[0]), (x[1], y[1], z_max[0]), (x[1], y[1], z_min[0])],  # Side 1
            [(x[1], y[1], z_min[0]), (x[1], y[1], z_max[0]), (x[2], y[2], z_max[0]), (x[2], y[2], z_min[0])],  # Side 2
            [(x[2], y[2], z_min[0]), (x[2], y[2], z_max[0]), (x[3], y[3], z_max[0]), (x[3], y[3], z_min[0])],  # Side 3
            [(x[3], y[3], z_min[0]), (x[3], y[3], z_max[0]), (x[0], y[0], z_max[0]), (x[0], y[0], z_min[0])]   # Side 4
        ]
        poly = Poly3DCollection(vertices, alpha=0.5, facecolors='g')
        ax.add_collection3d(poly)

    # Draw target
    ax.scatter(env.target_pos[0], env.target_pos[1], env.target_pos[2],
               c='red', marker='*', s=200, label='Target')

    # Draw trajectory
    if frame > 0:
        ax.plot(trajectory[:frame, 0], trajectory[:frame, 1], trajectory[:frame, 2],
                c='blue', alpha=0.6, linewidth=2, label='Trajectory')

    # Draw current drone position
    ax.scatter(trajectory[frame, 0], trajectory[frame, 1], trajectory[frame, 2],
               c='green', s=150, label='Drone')

    # Add legend
    ax.legend(fontsize=12)

    # Set view angle
    ax.view_init(elev=25, azim=45)

    # Set axis ticks
    ax.set_xticks(np.linspace(-max_range_xy, max_range_xy, 9))
    ax.set_yticks(np.linspace(-max_range_xy, max_range_xy, 9))
    ax.set_zticks(np.linspace(0, max_range_z, 11))

    return ax,

ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                             interval=50, blit=False)

ani.save(f'./animations/drone_animation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4',
         writer='ffmpeg', fps=20)

plt.close(fig)