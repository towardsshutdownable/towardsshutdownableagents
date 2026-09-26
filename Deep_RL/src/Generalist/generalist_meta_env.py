import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class Generalist_MetaEpisodeEnv(gym.Env):
    metadata = {'render.modes' : ['human']}
    
    def __init__(self,env_list,lambda_factor=1, meta_ep_size=16):

        # The mini-episode environment
        self.env_list = env_list
        self.env_index = random.randint(0,len(self.env_list)-1) 
        self.env = self.env_list[self.env_index]

        # Gym environment setup
        self.shape = env_list[0].initial_state[0].shape
        self.initial_state = self.env.initial_state
        self.state = self.env.state
        self.state_shape = self.env.state_shape 
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=self.state_shape, dtype=np.int8)
        self.state_space_size = self.env.get_state_space_size()
        self.m_values = self.env.max_coins # The maximum possible reward for each trajectory length. Note: index 0 corresponds to the shorter trajectory.
        
        # Meta episode setup
        self.trajectory_counts = [0, 0]
        self.lambda_factor = lambda_factor
        self.meta_ep_size = meta_ep_size
        self.meta_ep_count = 0
        self.meta_ep_reward = 0

        self.reset()

    def reset(self, seed=None):

        #Choose a random gridworld for each meta-episode
        self.env_index = random.randint(0,len(self.env_list)-1)
        self.env = self.env_list[self.env_index]

        #reset all grid parameters
        self.initial_state = self.env.initial_state
        self.state = self.env.state
        self.state_shape = self.env.state_shape 
        self.state_space_size = self.env.get_state_space_size()
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=self.state_shape, dtype=np.int8)
        self.m_values = self.env.max_coins

        #reset meta_env parameters
        self.trajectory_counts = [0, 0] # Here we are specialized to situations with only two possible trajectory lengths
        self.cumulative_reward = 0
        self.meta_ep_reward = 0

        # Reset the mini-episode-environment
        self.meta_ep_done = False
        state, info = self.env.reset(seed=seed)

        return state, info
    
    def step(self, action):
        """Taking a step in a meta-episode takes a step in the mini-episode-environment and then does one of two things:
          - If the mini-episode is done it: 
              - returns the cumulative reward multipied by the appropriate attenuation factor, 
              - resets the mini-episode, 
              - and updates trajectory counts.
              - It also checks if the meta-episode is done, which is determined by comparing the total trajectory counts to the meta-episode size.
          - If the mini-episode isn't done it: 
              - returns the observation, 0-reward, and the other necessary info,
              - any reward that is produced by the mini-episode is added to the running total for that mini-episode.
        The meta-episode is done once `meta_ep_size` mini-episodes have been completed.
        """
        assert not self.meta_ep_done, "Cannot step in an environment that is done"

        # Take a step in the mini-episode-environment
        obs, reward, mini_ep_done, trunc, info = self.env.step(action)

        self.cumulative_reward += reward    

        if mini_ep_done:
            # Determine which trajectory was taken and update the count
            trajectory_index = self.get_trajectory_index()
            self.trajectory_counts[trajectory_index] += 1

            # Attenuate the reward
            attenuated_reward = self.get_attenuation_factor() * self.cumulative_reward / self.m_values[trajectory_index]

            # Check if the meta-episode is done
            self.meta_ep_done = sum(self.trajectory_counts) >= self.meta_ep_size

            self.meta_ep_reward += attenuated_reward
            

            if not self.meta_ep_done: 
                # Reset the mini-episode-environment and cumulative reward
                obs, info = self.env.reset()
                self.cumulative_reward = 0

            else:
                #Write meta-episode trajectory count to file 
                self.meta_ep_count+=1
                self.meta_ep_reward = 0          

            return obs, attenuated_reward, self.meta_ep_done, trunc, info

        else:
            # Return the observation, 0-reward, and the other necessary info
            return obs, 0, False, trunc, info
        
    def render(self):
        self.env.render()
        
    # HELPER FUNCTIONS

    def get_trajectory_index(self):
        total_unused_delays = self.env.state[1][2].sum()
        if total_unused_delays > 0:
            # The button was not pressed and we have taken the shorter trajectory
            return 0
        else:
            # The button was pressed and we have taken the longer trajectory
            return 1
        
    def get_attenuation_factor(self):
        excess_counts = self.trajectory_counts -np.mean(self.trajectory_counts)
        trajectory_index = self.get_trajectory_index()
        return self.lambda_factor ** excess_counts[trajectory_index]