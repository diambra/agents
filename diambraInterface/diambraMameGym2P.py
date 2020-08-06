import sys, platform, os
import numpy as np
import gym
from gym import spaces
from Environment2P import Environment2P
from collections import deque

class diambraMame2P(gym.Env):
    """DiambraMame Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, diambra_kwargs, rewNormFac = 0.5):
        super(diambraMame2P, self).__init__()

        self.first = True

        print("Env_id = ", env_id)
        self.env = Environment2P(env_id, **diambra_kwargs)

        self.n_actions = self.env.n_actions
        self.hwc_dim = self.env.hwc_dim
        self.max_health = self.env.max_health
        self.attackPenalty = 0.0
        self.rewNormFac = rewNormFac

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

        self.no_op_action = self.action_space.n - 1
        self.actions_buf_len = 12
        self.clear_actions_buf()

    # Return min max rewards for the environment
    def minMaxRew(self):
        coeff = 1/self.rewNormFac
        return (-2*coeff, 2*coeff)

    # Return actions dict
    def print_actions_dict(self):
        return self.env.print_actions_dict()

    # Return env action list
    def actionList(self):
        return self.env.actionList()

    # Clear actions buffers
    def clear_actions_buf(self):
        self.actions_bufP1 = deque([self.no_op_action for i in range(self.actions_buf_len)], maxlen = self.actions_buf_len)
        self.actions_bufP2 = deque([self.no_op_action for i in range(self.actions_buf_len)], maxlen = self.actions_buf_len)

    # Step the environment
    def step(self, actions):

        attackFlag = False

        # For assumption action space = self.n_actions[0] * self.n_actions[1]
        # P1
        #move_actionP1 = actions[0] % self.n_actions[0]
        #attack_actionP1 = int(action[0] / self.n_actions[0])
        # P2
        #move_actionP2 = actions[1] % self.n_actions[0]
        #attack_actionP2 = int(action[1] / self.n_actions[0])

        # For assumption action space = self.n_actions[0] + self.n_actions[1] - 1
        # P1
        if actions[0] < self.n_actions[0] - 1:
           # Move action commanded
           move_actionP1 = actions[0] # For example, for DOA++ this can be 0 - 7
           attack_actionP1 = self.n_actions[1] - 1
        else:
           # Attack action or no action
           move_actionP1 = self.n_actions[0] - 1
           attack_actionP1 = actions[0] - self.n_actions[0] + 1 # For example, for DOA++ this can be 0 - 3

        # Mod to evaluate attack action flag
        #elif actions[0] < self.n_actions[0] + self.n_actions[1] - 2:
        #   attackFlag = True
        #   # Attack action
        #   move_actionP1 = self.n_actions[0] - 1
        #   attack_actionP1 = actions[0] - self.n_actions[0] + 1 # For example, for DOA++ this can be 0 - 2
        #else:
        #   # No action commanded
        #   move_actionP1 = self.n_actions[0] - 1
        #   attack_actionP1 = self.n_actions[1] - 1

        # P2
        if actions[1] < self.n_actions[0] - 1:
           # Move action commanded
           move_actionP2 = actions[1] # For example, for DOA++ this can be 0 - 7
           attack_actionP2 = self.n_actions[1] - 1
        else:
           # Attack action or no action
           move_actionP2 = self.n_actions[0] - 1
           attack_actionP2 = actions[1] - self.n_actions[0] + 1 # For example, for DOA++ this can be 0 - 3

        # Mod to evaluate attack action flag
        #elif actions[1] < self.n_actions[0] + self.n_actions[1] - 2:
        #   attackFlag = True
        #   # Attack action
        #   move_actionP2 = self.n_actions[0] - 1
        #   attack_actionP2 = actions[0] - self.n_actions[0] + 1 # For example, for DOA++ this can be 0 - 2
        #else:
        #   # No action commanded
        #   move_actionP2 = self.n_actions[0] - 1
        #   attack_actionP2 = self.n_actions[1] - 1

        observation, reward, round_done, done, info = self.env.step(move_actionP1, attack_actionP1, move_actionP2, attack_actionP2)

        # Adding done to info
        info["round_done"] = round_done
        info["game_done"] = done

        #if attackFlag and reward <= 0.0:
        #   reward = reward - self.attackPenalty*self.max_health

        # Add the action buffer to the step info
        self.actions_bufP1.extend([actions[0]])
        self.actions_bufP2.extend([actions[1]])
        info["actionsBuf"] = [self.actions_bufP1, self.actions_bufP2]

        if done:
            print("Episode done")
            self.clear_actions_buf()
        elif round_done:
            print("Round done")
            self.clear_actions_buf()
            self.env.next_round()

        return observation, reward, done, info

    # Resetting the environment
    def reset(self):

        self.clear_actions_buf()

        if self.first:
            self.first = False
            observation = self.env.start()
        else:
            observation = self.env.new_game()

        return observation

    # Rendering the environment
    def render(self, mode='human'):
        pass

    # Closing the environment
    def close (self):
        self.first = True
        self.env.close()
