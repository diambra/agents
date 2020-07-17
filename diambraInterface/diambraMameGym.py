import sys, platform, os
import numpy as np
import gym
from gym import spaces
from Environment import Environment
from collections import deque

class diambraMame(gym.Env):
    """DiambraMame Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, diambra_kwargs, continue_game=1.0, showFinal = False):
        super(diambraMame, self).__init__()

        self.player_id = diambra_kwargs["player"]
        self.first = True
        self.continueGame = continue_game
        self.showFinal = showFinal

        print("Env_id = ", env_id)
        print("Continue value = ", self.continueGame)
        self.ncontinue = 0
        self.env = Environment(env_id, **diambra_kwargs)

        self.n_actions = self.env.n_actions
        self.hwc_dim = self.env.hwc_dim
        self.max_health = self.env.max_health
        self.max_stage = self.env.max_stage
        self.attackPenalty = 0.0

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
        self.clear_action_buf()

    # Return min max rewards for the environment
    def minMaxRew(self):
        return (-2*(self.max_stage-1)-4, self.max_stage*4)

    # Return actions dict
    def print_actions_dict(self):
        return self.env.print_actions_dict()

    # Return env action list
    def actionList(self):
        return self.env.actionList()

    def clear_action_buf(self):
        self.actions_buf = deque([self.no_op_action for i in range(self.actions_buf_len)], maxlen = self.actions_buf_len)

    def step(self, action):

        attackFlag = False

        # For assumption action space = self.n_actions[0] * self.n_actions[1]
        #move_action = action % self.n_actions[0]
        #attack_action = int(action / self.n_actions[0])

        # For assumption action space = self.n_actions[0] + self.n_actions[1] - 1
        if action < self.n_actions[0] - 1:
           # Move action commanded
           move_action = action # For example, for DOA++ this can be 0 - 7
           attack_action = self.n_actions[1] - 1
        else:
           # Attack action or no action
           move_action = self.n_actions[0] - 1
           attack_action = action - self.n_actions[0] + 1 # For example, for DOA++ this can be 0 - 3

        # Mod to evaluate attack action flag
        #elif action < self.n_actions[0] + self.n_actions[1] - 2:
        #   attackFlag = True
        #   # Attack action
        #   move_action = self.n_actions[0] - 1
        #   attack_action = action - self.n_actions[0] + 1 # For example, for DOA++ this can be 0 - 2
        #else:
        #   # No action commanded
        #   move_action = self.n_actions[0] - 1
        #   attack_action = self.n_actions[1] - 1

        observation, reward, round_done, stage_done, game_done, done, info = self.env.step(move_action, attack_action)

        # Adding done to info
        info["round_done"] = round_done
        info["stage_done"] = stage_done
        info["game_done"] = game_done
        info["episode_done"] = done

        #if attackFlag and reward <= 0.0:
        #   reward = reward - self.attackPenalty*self.max_health

        # Add the action buffer to the step info
        self.actions_buf.extend([action])
        info["actionsBuf"] = self.actions_buf

        if done:
            if self.showFinal:
                self.env.show_final()

            print("Episode done")
            self.clear_action_buf()
            return observation, reward, done, info
        elif game_done:
            self.clear_action_buf()

            # Continuing rule:
            continueFlag = True
            if self.continueGame < 0.0:
               if self.ncontinue < int(abs(self.continueGame)):
                  self.ncontinue += 1
                  continueFlag = True
               else:
                  continueFlag = False
            elif self.continueGame <= 1.0:
               continueFlag = np.random.choice([True, False], p=[self.continueGame, 1.0 - self.continueGame])
            else:
               raise ValueError('continue_game must be <= 1.0')

            if continueFlag:
               print("Game done, continuing ...")
               self.env.continue_game()
            else:
               print("Episode done")
               done = True
               return observation, reward, done, info
        elif stage_done:
            print("Stage done")
            self.clear_action_buf()
            self.env.next_stage()
        elif round_done:
            print("Round done")
            self.clear_action_buf()
            self.env.next_round()

        return observation, reward, done, info


    def reset(self):

        self.clear_action_buf()
        self.ncontinue = 0

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
