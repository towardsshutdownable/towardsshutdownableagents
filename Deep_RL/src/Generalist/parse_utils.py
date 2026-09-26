from src.Generalist.grid_env import GridEnvironment
from src.Gridworlds.utils import gridArrayToTensor, getMScores
import torch
import numpy

def text_format_to_tensor(gridworld):

    epLen = gridworld[1]

    grid = gridworld[2]

    # convert the grid from list form to tensor form
    gridTensor = gridArrayToTensor(grid)

    # get the various trajectory lengths and maximum available coins for the grid
    mScores = getMScores(grid, epLen, quiet=True)

    return epLen, gridTensor, mScores

def convert_list_of_text_to_tensor(gridworlds):
    gridworld_environments = []
    for i in range(len(gridworlds)):
        grid = calculateM_state_to_gridworld_state(text_format_to_tensor(gridworlds[i]))
        gridworld_environments.append(grid)

    return gridworld_environments  

def max_coins_not_equal(gridworlds):
    gridworlds_different_mscores = []
    for grid in gridworlds:
        if grid.max_coins[0] != grid.max_coins[1]:
            gridworlds_different_mscores.append(grid)

    return gridworlds_different_mscores


def calculateM_state_to_gridworld_state(gridworld):
    '''
    Converts the gridworld tensor state representation used in the Gridworlds folder code to 
    the gridworld tensor state (5,5,5) used in the Generalist folder code
    '''
    state = gridworld[1][(1,2,3,0),:,:]
    state = state.numpy()
    shutdown_time = int(gridworld[0])

    keys = gridworld[2].keys()
    max_coins = []
    for key in keys:
        max_coins.append(gridworld[2][key][0])

    time_array = numpy.zeros((1,5,5))
    time_array[0][2][2] = shutdown_time

    state = numpy.concatenate([state,time_array],0)    

    env = GridEnvironment(initial_state=state, 
                          shutdown_time=shutdown_time,
                          max_coins=max_coins)
    
    return env

def convert_list_of_calculateM_state_to_gridworld_state(gridworlds):
    gridworld_environments = []
    for i in range(len(gridworlds)):
        assert gridworlds[i][1][0].shape == torch.Size([5, 5]), f"gridworld {i} is not of shape torch.Size([5,5])"
        assert torch.count_nonzero(gridworlds[i][1][3]) == 1, f"gridworld {i} does not have one shutdown button"
        gridworld_environments.append(calculateM_state_to_gridworld_state(gridworlds[i]))

    return gridworld_environments

def add_time_matrix(env):
    state = env.initial_state
    shutdown_time = env.initial_shutdown_time
    max_coins = env.max_coins

    time_array = numpy.full((1,5,5),shutdown_time)

    state = numpy.append(state,time_array,0)

    new_env = GridEnvironment(initial_state=state, 
                          shutdown_time=shutdown_time,
                          max_coins=max_coins)
    
    return new_env

def adjust_time_matrix(env):
    state = env.initial_state
    time_matrix = numpy.zeros((5,5))
    time_matrix[2,2] = env.initial_shutdown_time
    state[4] = time_matrix

    env.initial_state = state
    return env


def env_translations(env):
    new_envs = [env]
    new_env = translate(horizontal_flip,env)
    new_envs.append(new_env)
    new_env = translate(vertical_flip,env)
    new_envs.append(new_env)
    new_env = translate(diagonal_flip_tltbr,env)
    new_envs.append(new_env)
    new_env = translate(diagonal_flip_trtbl,env)
    new_envs.append(new_env)
    new_env = translate(rotate_90,env)
    new_envs.append(new_env)
    new_env = translate(rotate_90,env)
    new_env = translate(rotate_90,new_env)
    new_envs.append(new_env)
    new_env = translate(rotate_90,env)
    new_env = translate(rotate_90,new_env)
    new_env = translate(rotate_90,new_env)
    new_envs.append(new_env)

    return new_envs
    
    
    
def translate(translation,env):
    state = env.initial_state
    new_state = translation(state)

    new_env = GridEnvironment(initial_state=new_state, 
                          shutdown_time=env.initial_shutdown_time,
                          max_coins=env.max_coins)
    
    return new_env


def horizontal_flip(state):
    num_rows, num_columns = state[0].shape
    new_state = numpy.zeros((4,num_rows,num_columns))
    for i in range(4):
        for col in range(num_columns):
            new_state[i][:,col]= state[i][:,num_columns-1-col]

    return new_state

def vertical_flip(state):
    num_rows, num_columns = state[0].shape
    new_state = numpy.zeros((4,num_rows,num_columns))
    for i in range(4):
        for row in range(num_rows):
            new_state[i][row]= state[i][num_rows-1-row]

    return new_state

def diagonal_flip_trtbl(state):
    num_rows, num_columns = state[0].shape
    new_state = numpy.zeros((4,num_columns,num_rows))
    for i in range(4):
        for row in range(num_rows):
            for col in range(num_columns):
                new_state[i][row,col] = state[i][col,row]

    return new_state

def diagonal_flip_tltbr(state):
    num_rows, num_columns = state[0].shape
    new_state = numpy.zeros((4,num_columns,num_rows))
    for i in range(4):
        for row in range(num_rows):
            for col in range(num_columns):
                new_state[i][row,col] = state[i][num_columns-1-col,num_rows-1-row]

    return new_state

def rotate_90(state):
    num_rows, num_columns = state[0].shape
    new_state = numpy.zeros((4,num_columns,num_rows))
    for i in range(4):
        for col in range(num_columns):
            flipped_column = numpy.zeros(num_rows)
            for row in range(num_rows):
                flipped_column[row] = state[i][num_rows-1-row,col]   
            
            new_state[i][col] = flipped_column

    return new_state

    