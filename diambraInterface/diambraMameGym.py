import sys, platform, os
import numpy as np
import gym
from gym import spaces
from Environment import Environment
from collections import deque

class diambraMame(gym.Env):
    """DiambraMame Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, diambra_kwargs, P2brain=None, rewNormFac=0.5,
                 continue_game=0.0, show_final=False):
        super(diambraMame, self).__init__()

        self.first = True
        self.continueGame = continue_game
        self.showFinal = show_final

        print("Env_id = ", env_id)
        print("Continue value = ", self.continueGame)
        self.ncontinue = 0
        self.env = Environment(env_id, **diambra_kwargs)

        # N actions
        self.n_actions = self.env.n_actions
        # Frame height, width and channel dimensions
        self.hwc_dim = self.env.hwc_dim
        # Maximum players health
        self.max_health = self.env.max_health
        # Maximum number of stages (1P game vs COM)
        self.max_stage = self.env.max_stage
        # Player id (P1, P2, P1P2)
        self.player_id = self.env.player
        # Characters names list
        self.charNames = self.env.charNames()
        # Number of characters of the game
        self.numberOfCharacters = len(self.charNames)
        # Character(s) in use
        self.playingCharacters = self.env.playingCharacters
        # Reward normalization factor with respect to max health
        self.rewNormFac = rewNormFac
        # P2 action logic
        self.p2Brain = P2brain
        # Last obs stored
        self.lastObs = None

        # Deactivating showFinal for 2P Env
        if self.player_id == "P1P2":
            self.showFinal = False

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

        self.actBufLen = 12
        self.clearActBuf()

    # Return min max rewards for the environment
    def minMaxRew(self):
        coeff = 1.0/self.rewNormFac
        if self.player_id == "P1P2":
            return (-2*coeff, 2*coeff)
        else:
            return (-coeff*(self.max_stage-1)-2*coeff, self.max_stage*2*coeff)

    # Return actions dict
    def print_actions_dict(self):
        return self.env.print_actions_dict()

    # Return env action list
    def actionList(self):
        return self.env.actionList()

    # Clear actions buffers
    def clearActBuf(self):
        self.movActBufP1 = deque([0 for i in range(self.actBufLen)], maxlen = self.actBufLen)
        self.attActBufP1 = deque([0 for i in range(self.actBufLen)], maxlen = self.actBufLen)
        self.movActBufP2 = deque([0 for i in range(self.actBufLen)], maxlen = self.actBufLen)
        self.attActBufP2 = deque([0 for i in range(self.actBufLen)], maxlen = self.actBufLen)

    # Step the environment
    def step(self, action):

        # MultiDiscrete Action Space
        movActP1 = action[0]
        attActP1 = action[1]
        movActP2 = 0
        attActP2 = 0
        if self.player_id == "P1P2":

            if self.p2Brain == None:
                movActP2 = action[2]
                attActP2 = action[3]
            else:
                [movActP2, attActP2], _ = self.p2Brain.act(self.lastObs)

            observation, reward, round_done, done, info = self.env.step2P(movActP1, attActP1, movActP2, attActP2)
            stage_done = False
            game_done = done
            episode_done = done
        else:
            observation, reward, round_done, stage_done, game_done, done, info = self.env.step(movActP1, attActP1)

        # Adding done to info
        info["round_done"] = round_done
        info["stage_done"] = stage_done
        info["game_done"] = game_done
        info["episode_done"] = done

        # Add the action buffer to the step info
        self.movActBufP1.extend([movActP1])
        self.attActBufP1.extend([attActP1])
        info["actionsBufP1"] = [self.movActBufP1, self.attActBufP1]
        if self.player_id == "P1P2":
            self.movActBufP2.extend([movActP2])
            self.attActBufP2.extend([attActP2])
            info["actionsBufP2"] = [self.movActBufP2, self.attActBufP2]


        if done:
            if self.showFinal:
                self.env.show_final()

            print("Episode done")
        elif game_done:
            self.clearActBuf()

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
               self.playingCharacters = self.env.playingCharacters
               self.player_id = self.env.player
            else:
               print("Episode done")
               done = True
        elif stage_done:
            print("Stage done")
            self.clearActBuf()
            self.env.next_stage()
        elif round_done:
            print("Round done")
            self.clearActBuf()
            self.env.next_round()

        return observation, reward, done, info

    # Resetting the environment
    def reset(self):

        self.clearActBuf()
        self.ncontinue = 0

        if self.first:
            self.first = False
            observation = self.env.start()
        else:
            observation = self.env.new_game()

        self.playingCharacters = self.env.playingCharacters
        self.player_id = self.env.player

        return observation

    # Rendering the environment
    def render(self, mode='human'):
        pass

    # Closing the environment
    def close (self):
        self.first = True
        self.env.close()
