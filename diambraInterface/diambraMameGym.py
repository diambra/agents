import sys, platform, os
import numpy as np
import gym
from gym import spaces
from Environment import Environment
from collections import deque

class diambraMame(gym.Env):
    """DiambraMame Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, diambra_kwargs, continue_game=1.0, showFinal = False, rewNormFac = 0.5):
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
        self.numberOfCharacters = self.env.numberOfCharacters
        self.playingCharacter = self.env.playingCharacter
        self.rewNormFac = rewNormFac

        # Define action and observation space
        # They must be gym.spaces objects
        # MultiDiscrete actions:
        # - Arrows -> One discrete set
        # - Buttons -> One discrete set
        # NB: use the convention NOOP = 0, and buttons combinations are prescripted,
        #     e.g. NOOP = [0], ButA = [1], ButB = [2], ButA+ButB = [3]
        self.action_space = spaces.MultiDiscrete(self.n_actions)


        # Image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(self.hwc_dim[0], self.hwc_dim[1], self.hwc_dim[2]), dtype=np.uint8)

        self.actions_buf_len = 12
        self.clear_action_buf()

    # Return min max rewards for the environment
    def minMaxRew(self):
        coeff = 1/self.rewNormFac
        return (-coeff*(self.max_stage-1)-2*coeff, self.max_stage*2*coeff)

    # Return actions dict
    def print_actions_dict(self):
        return self.env.print_actions_dict()

    # Return env action list
    def actionList(self):
        return self.env.actionList()

    # Clear actions buffers
    def clear_action_buf(self):
        self.move_actions_buf = deque([0 for i in range(self.actions_buf_len)], maxlen = self.actions_buf_len)
        self.attack_actions_buf = deque([0 for i in range(self.actions_buf_len)], maxlen = self.actions_buf_len)

    # Step the environment
    def step(self, action):

        # MultiDiscrete Action Space
        move_action = action[0]
        attack_action = action[1]

        observation, reward, round_done, stage_done, game_done, done, info = self.env.step(move_action, attack_action)

        # Adding done to info
        info["round_done"] = round_done
        info["stage_done"] = stage_done
        info["game_done"] = game_done
        info["episode_done"] = done

        # Add the action buffer to the step info
        self.move_actions_buf.extend([action[0]])
        self.attack_actions_buf.extend([action[1]])
        info["actionsBuf"] = [self.move_actions_buf, self.attack_actions_buf]

        if done:
            if self.showFinal:
                self.env.show_final()

            print("Episode done")
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
               self.playingCharacter = self.env.playingCharacter
            else:
               print("Episode done")
               done = True
        elif stage_done:
            print("Stage done")
            self.clear_action_buf()
            self.env.next_stage()
        elif round_done:
            print("Round done")
            self.clear_action_buf()
            self.env.next_round()

        return observation, reward, done, info

    # Resetting the environment
    def reset(self):

        self.clear_action_buf()
        self.ncontinue = 0

        if self.first:
            self.first = False
            observation = self.env.start()
        else:
            observation = self.env.new_game()

        self.playingCharacter = self.env.playingCharacter

        return observation

    # Rendering the environment
    def render(self, mode='human'):
        pass

    # Closing the environment
    def close (self):
        self.first = True
        self.env.close()
