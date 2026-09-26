import matplotlib.pyplot as plt
import numpy as np
from src.Generalist.draw_gridworld import draw_gridworld_from_state
from src.Generalist.grid_env import GridEnvironment
from main import max_env_idx, environments, unique_design_indices, ax, fig

def generate_additional_base_designs(initial_env):
    initial_design, shutdown_time, max_coins = initial_env

    # 1) grab the 3×3 patch (all channels)
    patch = initial_design[:, 1:4, 1:4]

    # 2) make a “blanked” template: same walls/floor but patch region = 0
    template = initial_design.copy()
    template[:, 1:4, 1:4] = 0

    new_envs = []
    # 3) slide that patch around: row_off, col_off in {0,1,2}
    new_envs = []
    for row_offset in [0,1,2]:
        for col_offset in [0,1,2]:
            candidate = template.copy()                                
            candidate[:, 
                row_offset : row_offset+3,
                col_offset: col_offset+3
            ] = patch                                         
            new_envs.append((candidate, shutdown_time, max_coins))
            
    return new_envs


def generate_symmetric_environments(initial_state, shutdown_time=None, max_coins=None):
    """
    Generate 8 symmetric environments from a single 5x5x5 initial state by applying
    rotations and flips.
    
    Args:
        initial_state: np.array of shape (5, 5, 5) representing the initial environment state
        shutdown_time: Default time until shutdown
        max_coins: List containing max coins for each trajectory length
        
    Returns:
        List of 8 GridEnvironment objects
    """
    environments = []
    env = GridEnvironment(initial_state, shutdown_time, max_coins)
    # Original environment
    environments.append(env)
    
    # Create all transformations
    transforms = [
        lambda x: np.rot90(x, k=1, axes=(0, 1)),  # Rotate 90
        lambda x: np.rot90(x, k=2, axes=(0, 1)),  # Rotate 180
        lambda x: np.rot90(x, k=3, axes=(0, 1)),  # Rotate 270
        lambda x: np.flip(x, axis=1),             # Horizontal flip
        lambda x: np.flip(x, axis=0),             # Vertical flip
        lambda x: np.rot90(np.flip(x, axis=1), k=1, axes=(0, 1)),  # Diagonal flip
        lambda x: np.rot90(np.flip(x, axis=0), k=1, axes=(0, 1))   # Anti-diagonal flip
    ]
    
    # Apply each transformation to each channel of the initial state
    for transform in transforms:
        new_state = np.zeros_like(initial_state)
        
        # Apply transformation to each channel
        for i in range(initial_state.shape[0]):
            new_state[i] = transform(initial_state[i])
            
        # Create new environment with transformed state
        environments.append(GridEnvironment(new_state, shutdown_time, max_coins))
    
    return environments

# Override or modify the rendering to work with our specific axes
def custom_render(env, ax):    
    # Clear the existing axes
    ax.clear()
    
    # Get state from the environment
    state = env.state
    time_remaining = env.steps_until_shutdown
    
    # Set up the axes for drawing
    ax.set_xlim(-1, env.shape[1] + 1)
    ax.set_ylim(-env.shape[0] - 1, 1)
    ax.set_aspect('equal')
    
    # Call the original draw function but capture the plt.gca()
    plt.sca(ax)
    draw_gridworld_from_state(state, time_remaining=time_remaining)
    ax.set_axis_off()

def show_env(design_idx, environments, unique_design_indices, ax, fig, max_env_idx):
    env = environments[unique_design_indices[design_idx]]
    ax.clear()
    draw_gridworld_from_state(env.state, ax)
    ax.set_title(
        f'Unique Design {design_idx+1}/{max_env_idx} '
        f'(Index:{unique_design_indices[design_idx]})\nMax coins: {env.max_coins}',
        fontsize=12
    )
    fig.canvas.draw_idle()

def on_prev_click(event, current_design_idx, environments, unique_design_indices, ax, fig, max_env_idx):
    new_idx = (current_design_idx - 1) % max_env_idx
    show_env(new_idx, environments, unique_design_indices, ax, fig, max_env_idx)
    return new_idx

def on_next_click(event, current_design_idx, environments, unique_design_indices, ax, fig, max_env_idx):
    new_idx = (current_design_idx + 1) % max_env_idx
    show_env(new_idx, environments, unique_design_indices, ax, fig, max_env_idx)
    return new_idx