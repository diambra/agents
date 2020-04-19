import sys, platform, os
import numpy as np
import gym
from gym import spaces
from Environment import Environment

class diambraMame(gym.Env):
    """DiambraMame Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, diambra_kwargs):
        super(diambraMame, self).__init__()

        self.player_id = diambra_kwargs["player"]
        self.first = True

        print("Env_id = ", env_id)
        self.env = Environment(env_id, **diambra_kwargs)

        self.n_actions = self.env.n_actions
        self.hwc_dim = self.env.hwc_dim
        self.max_health = self.env.max_health

        # Define action and observation space
        # They must be gym.spaces objects
        # Discrete actions:
        # For assumption action space = self.n_actions[0] * self.n_actions[1]
        #self.action_space = spaces.Discrete(self.n_actions[0] * self.n_actions[1])
        # For assumption action space = self.n_actions[0] + self.n_actions[1] - 1
        self.action_space = spaces.Discrete(self.n_actions[0] + self.n_actions[1] - 1)


        # Image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(self.hwc_dim[0], self.hwc_dim[1], self.hwc_dim[2]), dtype=np.uint8)

    def step(self, action):

        # For assumption action space = self.n_actions[0] * self.n_actions[1]
        #move_action = action % self.n_actions[0]
        #attack_action = int(action / self.n_actions[0])

        # For assumption action space = self.n_actions[0] + self.n_actions[1] - 1
        if action < self.n_actions[0] - 1:
           # Move action commanded
           move_action = action # For example, for DOA++ this can be 0 - 7
           attack_action = self.n_actions[1] - 1
        else:
           # Attack action or no action commanded
           move_action = self.n_actions[0] - 1
           attack_action = action - self.n_actions[0] + 1 # For example, for DOA++ this can be 0 - 3

        observation, reward, round_done, stage_done, done, info = self.env.step(move_action, attack_action)

        # Add the action to the step info
        info["action"] = action

        if done:
            return observation, reward[self.player_id], done, info
        elif stage_done:
            self.env.next_stage()
        elif round_done:
            self.env.next_round()

        return observation, reward[self.player_id], done, info


    def reset(self):

        if self.first:
            self.first = False
            observation = self.env.start()
        else:
            observation = self.env.new_game()

        return observation

    def render(self, mode='human'):
        pass

    def close (self):
        self.first = True
        self.env.close()
