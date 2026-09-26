import pickle
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import numpy as np
from src.Generalist.draw_gridworld import draw_gridworld_from_state
from random import randint
from src.Generalist.grid_env import GridEnvironment
import numpy as np
from world_design import base_designs_red_orange_envs, test_set, master_set

gridworlds = [0, 16, 24, 32, 40, 64, 88, 96, 112, 128, 136, 160, 184, 8, 40, 64, 112, 128, 168, 144, 216, 288, 360, 576, 416, 424]

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

def calculateM_state_to_gridworld_state(gridworld):
    '''
    Converts the gridworld tensor state representation used in the Gridworlds folder code to 
    the gridworld tensor state (5,5,5) used in the Generalist folder code
    '''
    state = gridworld[1][(1,2,3,0),:,:] # <- THIS CHANGES THE ORDER OF THE 5X5 MATRICES i.e index 1 matrix should be index 0 (i.e 2nd -> 1st)
    state = state.numpy()
    shutdown_time = int(gridworld[0]) # <- THIS GETS THE SHUTDOWN TIME, THE 4 IN THE EXAMPLE YOU SENT

    keys = gridworld[2].keys() # <- THIS GETTS THE KEYS IN THE DICT
    max_coins = []
    for key in keys:
        max_coins.append(gridworld[2][key][0]) # <- THIS GETS THE 4 & 9

    time_array = np.zeros((1,5,5)) # <- THIS CREATES 5X5 FULL OF ZEROES
    time_array[0][2][2] = shutdown_time # <- THIS PUT TIME_UNTIL_SHUTDOWN IN THE CENTRE 

    state = np.concatenate([state,time_array],0) # <- THIS PUTS TIME_UNTIL_SHUTDOWN MATRIX AT THE END
    
    return state, shutdown_time, max_coins

environments = []

processed_envs_filename = 'world_builder/worlds/extracted_envs.pkl'
with open(processed_envs_filename, 'rb') as f:
    newly_loaded_envs = pickle.load(f)

master_set.extend(newly_loaded_envs)

for idx in range(len(master_set)):
    if idx in [*range(25,50)]:
        initial_design, shutdown_time, max_coins = calculateM_state_to_gridworld_state(master_set[idx])
        #environments.append(GridEnvironment(initial_design, shutdown_time, max_coins))
        env_transformed_list = generate_symmetric_environments(initial_design, shutdown_time, max_coins)
        for transformed_env in env_transformed_list:
                environments.append(transformed_env)
        '''env_transformed_list = generate_symmetric_environments(initial_design, shutdown_time, max_coins)
        for transformed_env in env_transformed_list:
                environments.append(transformed_env)'''
        '''new_bases = generate_additional_base_designs(master_set[idx])
        for new_base_env in new_bases:
            initial_design, shutdown_time, max_coins = new_base_env
            environments.append(GridEnvironment(initial_design, shutdown_time, max_coins))
        for new_base_env in new_bases:
            env_transformed_list = generate_symmetric_environments(new_base_env[0], new_base_env[1], new_base_env[2])
            for transformed_env in env_transformed_list:
                    environments.append(transformed_env)'''
    else:
        initial_design, shutdown_time, max_coins = master_set[idx]
        print(master_set[idx])
        #environments.append(GridEnvironment(initial_design, shutdown_time, max_coins))
        env_transformed_list = generate_symmetric_environments(initial_design, shutdown_time, max_coins)
        for transformed_env in env_transformed_list:
                environments.append(transformed_env)

# Create a directory to store the pickled files if it doesn't exist
os.makedirs("src/Gridworlds", exist_ok=True)

# Save to pickle file
output_path = "world_builder/worlds/master_set_400.pkl"
with open(output_path, "wb") as f:
    pickle.dump(environments, f)

print(f"Saved {len(environments)} environments to {output_path}")

# ---- DRAW ENVIRONMENTS -----

# Set up the figure and axes
fig = plt.figure(figsize=(8, 9))  # Make it taller to accommodate buttons
ax = plt.axes([0.1, 0.2, 0.8, 0.7])  # Main axes for the gridworld
plt.subplots_adjust(bottom=0.15)  # Make room for buttons

# Create a list of indices for unique environment designs (every 8th environment)
unique_design_indices = list(range(0, len(environments)))
max_env_idx = len(unique_design_indices)
current_design_idx = 0

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

def show_env(design_idx):
    global current_design_idx
    current_design_idx = design_idx
    
    # Get the actual environment index
    env_idx = unique_design_indices[design_idx]
    
    # Clear and redraw the main plot
    plt.sca(ax)
    env = environments[env_idx]
    env.reset()
    
    # Use our custom rendering function
    custom_render(env, ax)
    
    # Set title
    ax.set_title(f'Unique Design {design_idx+1}/{max_env_idx} (Idx:{env_idx})\nMax coins: {env.max_coins}', fontsize=12)
    
    # Make sure the figure is updated
    fig.canvas.draw_idle()

# Create button axes and button functionality
ax_prev = plt.axes([0.3, 0.05, 0.15, 0.075])
ax_next = plt.axes([0.55, 0.05, 0.15, 0.075])

def on_prev_click(event):
    show_env((current_design_idx - 1) % max_env_idx)

def on_next_click(event):
    show_env((current_design_idx + 1) % max_env_idx)

btn_prev = Button(ax_prev, 'Previous')
btn_prev.on_clicked(on_prev_click)

btn_next = Button(ax_next, 'Next')
btn_next.on_clicked(on_next_click)

# Show the first environment
show_env(current_design_idx)
plt.show()