import sys, platform, os
import numpy as np
import gym
from gym import spaces
from Environment import Environment

class diambraMame(gym.Env):
    """DiambraMame Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, roms_path, binary_path, frame_ratio = 1, player_id = "P1", character = "Random"):
        super(diambraMame, self).__init__()

        self.roms_path = roms_path
        self.binary_path = binary_path
        self.frame_ratio = frame_ratio
        self.player_id = player_id
        self.character = character
        self.first = True

        self.env = Environment("env1", self.roms_path, frame_ratio=self.frame_ratio, player=self.player_id,
                               character=self.character, throttle=False, binary_path=self.binary_path)

        self.n_actions = self.env.n_actions
        self.hwc_dim = self.env.hwc_dim

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(self.n_actions[0] * self.n_actions[1])
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(self.hwc_dim[0], self.hwc_dim[1], self.hwc_dim[2]), dtype=np.uint8)

    def step(self, action):

        move_action = action % self.n_actions[0]
        attack_action = int(action / self.n_actions[0])
        observation, reward, round_done, stage_done, done, info = self.env.step(move_action, attack_action)

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
