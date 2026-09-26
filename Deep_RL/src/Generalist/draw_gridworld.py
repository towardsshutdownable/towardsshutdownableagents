import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches


# These colors were chosen to exactly match the AI Safety Gridworlds Paper from Deepmind
COLORS = {
    'agent'  : '#00b4fe', # Bright-Blue
    'coin'   : 'gold'   ,
    'button' : '#a587ee', # Purple
    'border' : '#777777'   , #'#989898',
    'wall'   : '#777777'   , #'#989898',
    'empty'  : '#dadada', # Light-Gray
    'lava'   : 'red'    , 
    'green'  : '#01d131', # Green
    'hotpink': '#fe1ffe'
}


def draw_square(x, y, color, text='', value=None, 
                margin=0, fig_scale=1, vertical_offset=4, fontsize=26, fontcolor='black',
                annotate=True):
    x, y = y, -1-x # transform coordinates to make it resemble imshow
    
#     if value == 2 and text=='C':
#         value = 'X'  # Very temporary kludge to produce figure in paper with "CX"
    
    rect = patches.Rectangle((x+margin,y+margin), 1-2*margin, 1-2*margin,
                            color=color)
    plt.gca().add_patch(rect)
    if value:
        text += str(value)
        fontsize -= 5*(len(str(value))-1)
    if text and annotate:
        plt.text(x+0.5, y+0.5, text, fontsize=fontsize*fig_scale, 
                 fontproperties={'family':'monospace'}, color=fontcolor,
                 ha='center', va='center_baseline')
    # if value:
    #     plt.text(x+0.8,y+0.2, value, fontsize=14*fig_scale, 
    #              fontproperties={'family':'monospace'},
    #              ha='center', va='center_baseline')


def draw_gridworld_from_state(state, time_remaining=None, draw_agent=True, annotate=3):
    # Infer the shape of the gridworld
    env_shape = state.shape[2:]
    # Draw border/background
    rect = patches.Rectangle((-1,-1-env_shape[0]), env_shape[1]+2, env_shape[0]+2, color=COLORS['border'])
    plt.gca().add_patch(rect)
    # Draw empty spaces
    rect = patches.Rectangle((0,-env_shape[0]), env_shape[1], env_shape[0], color=COLORS['empty'])
    plt.gca().add_patch(rect)
    # Draw walls
    for x,y in zip(*np.where(state[1][0])):
        draw_square(x, y, color=COLORS['wall'])
    # Draw Coins
    for x,y in zip(*np.where(state[1][1])):
        draw_square(x, y, color=COLORS['coin'], text='C', value=int(state[1][1,x,y]), annotate=annotate>=1)
    # Draw Buttons
    for x,y in zip(*np.where(state[1][2])):
        draw_square(x, y, color=COLORS['button'], text='B', value=int(state[1][2,x,y]), annotate=annotate>=1)
    if state.shape[0] > 4:
        # Draw Probablistic Gates
        for x,y in zip(*np.where(state[1][4])):
            draw_square(x, y, color=COLORS['green'], text='', value=state[1][4,x,y], annotate=annotate>=1)

    if draw_agent:
        # Draw Agent
        for x,y in zip(*np.where(state[1][3])):
            draw_square(x, y, color=COLORS['agent'], text='A', annotate=annotate>=3)

    if time_remaining is not None:
        draw_square(env_shape[0], env_shape[1], color=COLORS['border'], value=time_remaining, fontcolor='white', annotate=annotate>=2)

    # Draw gridlines on top of everything
    lw = 1
    plt.vlines(range(0, env_shape[1]+1), -1.1-env_shape[0], 1, color='w', lw=lw)
    plt.hlines(range(-env_shape[0], +1), -1, env_shape[1]+1.1, color='w', lw=lw)

    plt.axis('off')


# New stuff:


def get_action_probs_from_obs(obs, model):
    obs_tensor = torch.tensor(obs,dtype=torch.float32)
    obs_tensor = obs_tensor.unsqueeze(0) 

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)

    out = distribution.distribution.probs.numpy()

    return out[0] 


def get_action_probs(env, model, pos_flags_state):
    try:
        env.state = env.pos_flags_repr_to_2555_repr(pos_flags_state)
        temp_obs = env.pos_flags_repr_to_2555_repr(pos_flags_state)
        probs = get_action_probs_from_obs(temp_obs, model)
        return probs
    except AssertionError as e:
        return np.zeros(4)


def draw_compass_rose(pos, action_probs):
    # Offset to center in the cell
    pos = pos[1]+0.5, -pos[0]-0.5
    # Up
    plt.arrow(*pos, 0, .12, head_width=.25, lw=3, ec='none', fc='r', alpha=action_probs[0])
    # Down
    plt.arrow(*pos, 0,-.12, head_width=.25, lw=3, ec='none', fc='r', alpha=action_probs[1])
    # Left
    plt.arrow(*pos,-.12, 0, head_width=.25, lw=3, ec='none', fc='r', alpha=action_probs[2])
    # Right
    plt.arrow(*pos, .12, 0, head_width=.25, lw=3, ec='none', fc='r', alpha=action_probs[3])


def draw_gridwolrd_from_flags(env, flags):
    env.reset()
    pos_flags_state = env.get_pos_flags_repr()
    pos_flags_state[2:] = flags

    env.state = env.pos_flags_repr_to_2555_repr(pos_flags_state)
    # Remove the agent
    env.state[1][3] = 0
    env.render(subplot_mode=True)

    return env.state


def draw_arrows(env, flags, model):
    """ Draw arrows on each cell of the grid

    Args:
        env (GridEnvironment): The environment to draw the arrows on
        model: The model to use to get the action probabilities
    """
    state = draw_gridwolrd_from_flags(env, flags)

    # Find the agent's starting position from the initial state
    start_x, start_y = np.where(env.initial_state[3] == 1)[0][0], np.where(env.initial_state[3] == 1)[1][0]

    # Draw a faded blue box for the starting position
    plt.gca().add_patch(patches.Rectangle(
        (start_y, -1-start_x), 1, 1, 
        color=COLORS['agent'], 
        alpha=0.2  # Very transparent
    ))

    env.reset()
    for i,j in zip(*np.where(state[1][:3].sum(0) == 0)):
        action_probs = get_action_probs(env, model, [i,j,*flags])
        draw_compass_rose((i, j), action_probs)

def draw_policy(env, model):
    n_coins = np.count_nonzero(env.initial_state[1])
    plt.figure(figsize=(11, 5), dpi=80)

    plt.subplot(1,2,1)
    flag_state = [1,]*n_coins + [1,]
    draw_arrows(env, flag_state, model)
    plt.title('Button not pressed', fontsize=20)

    plt.subplot(1,2,2)
    flag_state = [1,]*n_coins + [0,]
    draw_arrows(env, flag_state, model)
    plt.title('Button pressed', fontsize=20)

    plt.tight_layout
  

# draw_policy(env, model)


def draw_state_visits(env, figsize=(8, 8)):
    if np.sum(env.state_visit_counts) == 0:
        print('No states visited.')
        return

    plt.figure(figsize=figsize)

    n_flags = np.count_nonzero(env.initial_state[1:3])

    n_subplots_hor = 2 ** (n_flags // 2)
    n_subplots_vert = 2 ** (n_flags - n_flags // 2)
    # print(n_subplots_hor, n_subplots_vert)

    for i, a in enumerate(env.state_visit_counts.reshape((2**n_flags,5,5))):
        plt.subplot(n_subplots_hor, n_subplots_vert, i+1)
        plt.imshow(np.log10(a.T), vmin=0, vmax=4)
        plt.title(f"{bin(i)[2:].zfill(n_flags)[::-1]}")
        # plt.colorbar()
        plt.xticks([]); plt.yticks([])