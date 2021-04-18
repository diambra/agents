import sys, platform, os
import numpy as np
import gym
from gym import spaces
from Environment import Environment
from collections import deque
from policies import P2ToP1AddObsMove

class diambraMame(gym.Env):
    """DiambraMame Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, diambra_kwargs, P2brain=None, rewNormFac=0.5,
                 continue_game=0.0, show_final=False, gamePads=[None, None],
                 actionSpace=["multiDiscrete", "multiDiscrete"],
                 attackButCombinations=[True, True]):
        super(diambraMame, self).__init__()

        self.first = True
        self.continueGame = continue_game
        self.showFinal = show_final
        self.actionSpace = actionSpace
        self.attackButCombinations=attackButCombinations

        print("Env_id = {}".format(env_id))
        print("Continue value = {}".format(self.continueGame))
        print("Action Spaces = {}".format(self.actionSpace))
        print("Use attack buttons combinations = {}".format(self.attackButCombinations))

        self.ncontinue = 0
        self.env = Environment(env_id, diambra_kwargs).getEnv()

        # N actions
        self.n_actions = [self.env.n_actions_butComb, self.env.n_actions_butComb]
        for idx in range(2):
            if not self.attackButCombinations[idx]:
                self.n_actions[idx] = self.env.n_actions_noButComb
        # Frame height, width and channel dimensions
        self.hwc_dim = self.env.hwc_dim
        # Maximum players health
        self.max_health = self.env.max_health
        # Maximum number of stages (1P game vs COM)
        self.max_stage = self.env.max_stage
        # Player Side (P1, P2, P1P2)
        self.playerSide = self.env.player
        # Player Id (P1->0, P2->1, P1P2->None)
        self.playerId = self.env.playerId
        # Characters names list
        self.charNames = self.env.charNames()
        # Number of characters of the game
        self.numberOfCharacters = len(self.charNames)
        # Character(s) in use
        self.playingCharacters = self.env.playingCharacters
        # Difficulty
        self.difficulty = self.env.difficulty
        # Reward normalization factor with respect to max health
        self.rewNormFac = rewNormFac

        # P2 action logic (for AIvsHUM and AIvsAI training)
        self.p2Brain = P2brain
        if self.p2Brain != None:
            self.p2Brain.initialize(self.env.actionList())

            # If p2 action logic is gamepad, add it to self.gamepads (for char selection)
            # Check action space is prescribed as "multiDiscrete"
            if self.p2Brain.id == "gamepad":
                gamePads[1] = self.p2Brain
                if self.actionsSpace[1] != "multiDiscrete":
                    raise Exception("Action Space for P2 must be \"multiDiscrete\" when using gamePad")
                if not self.attackButCombinations[1]:
                    raise Exception("Use attack buttons combinations for P2 must be \"True\" when using gamePad")

        # Last obs stored (for AIvsAI training)
        self.lastObs = None

        # Gamepads (for char selection)
        self.gamePads = gamePads
        gamepadNum = 0
        for idx in range(2):
            if self.gamePads[idx] != None:
                self.gamePads[idx].initialize(self.env.actionList(), gamepadNum=gamepadNum)
                gamepadNum += 1

        # Define action and observation space
        # They must be gym.spaces objects
        # We check only first element of self.actionSpace since for now only 1P
        # training is supported and here defining the action space is for training

        if self.actionSpace[0] == "multiDiscrete":
            # MultiDiscrete actions:
            # - Arrows -> One discrete set
            # - Buttons -> One discrete set
            # NB: use the convention NOOP = 0, and buttons combinations
            #     can be prescripted:
            #     e.g. NOOP = [0], ButA = [1], ButB = [2], ButA+ButB = [3]
            #     or ignored:
            #     e.g. NOOP = [0], ButA = [1], ButB = [2]
            self.action_space = spaces.MultiDiscrete(self.n_actions[0])
            print("Using MultiDiscrete action space")
        elif self.actionSpace[0] == "discrete":
            # Discrete actions:
            # - Arrows U Buttons -> One discrete set
            # NB: use the convention NOOP = 0, and buttons combinations
            #     can be prescripted:
            #     e.g. NOOP = [0], ButA = [1], ButB = [2], ButA+ButB = [3]
            #     or ignored:
            #     e.g. NOOP = [0], ButA = [1], ButB = [2]
            self.action_space = spaces.Discrete(self.n_actions[0][0] + self.n_actions[0][1] - 1)
            print("Using Discrete action space")
        else:
            raise Exception("Not recognized action space: {}".format(self.actionSpace[0]))

        # Image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(self.hwc_dim[0], self.hwc_dim[1], self.hwc_dim[2]), dtype=np.uint8)

        # Saving both action spaces
        self.action_spaces = [None, None]
        for idx in range(2):
            if self.actionSpace[idx] == "multiDiscrete":
                self.action_spaces[idx] = spaces.MultiDiscrete(self.n_actions[idx])
            else:
                self.action_spaces[idx] = spaces.Discrete(self.n_actions[idx][0] + self.n_actions[idx][1] - 1)

        self.actBufLen = 12
        self.clearActBuf()

    # Return min max rewards for the environment
    def minMaxRew(self):
        coeff = 1.0/self.rewNormFac
        if self.env.inputPlayer == "P1P2":
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

    # Discrete to multidiscrete action conversion
    def discreteToMultiDiscreteAction(self, action):

        movAct = 0
        attAct = 0

        if action <= self.n_actions[0][0] - 1:
            # Move action or no action
            movAct = action # For example, for DOA++ this can be 0 - 8
        else:
            # Attack action
            attAct = action - self.n_actions[0][0] + 1 # For example, for DOA++ this can be 1 - 7

        return movAct, attAct

    # Step the environment
    def step(self, action):

        # Actions initialization
        movActP1 = 0
        attActP1 = 0
        movActP2 = 0
        attActP2 = 0

        # Defining move and attack actions P1/P2 as a function of actionSpace
        if self.actionSpace[0] == "multiDiscrete": # P1 MultiDiscrete Action Space

            # P1
            movActP1 = action[0]
            attActP1 = action[1]

            # P2
            if self.playerSide == "P1P2":

                if self.actionSpace[1] == "multiDiscrete": # P2 MultiDiscrete Action Space

                    if self.p2Brain == None:
                        movActP2 = action[2]
                        attActP2 = action[3]
                    else:
                        if self.p2Brain.id == "rl":
                            self.lastObs[:,:,-1] = P2ToP1AddObsMove(self.lastObs[:,:,-1])
                        [movActP2, attActP2], _ = self.p2Brain.act(self.lastObs)

                else: # P2 Discrete Action Space

                    if self.p2Brain == None:
                        movActP2, attActP2 = self.discreteToMultiDiscreteAction(action[2])
                    else:
                        if self.p2Brain.id == "rl":
                            self.lastObs[:,:,-1] = P2ToP1AddObsMove(self.lastObs[:,:,-1])
                        brainActions, _ = self.p2Brain.act(self.lastObs)
                        movActP2, attActP2 = self.discreteToMultiDiscreteAction(brainActions)

        else: # P1 Discrete Action Space

            if self.playerSide != "P1P2":

                # P1
                # Discrete to multidiscrete conversion
                movActP1, attActP1 = self.discreteToMultiDiscreteAction(action)

            else:

                # P1
                # Discrete to multidiscrete conversion
                movActP1, attActP1 = self.discreteToMultiDiscreteAction(action[0])

                # P2
                if self.actionSpace[1] == "multiDiscrete": # P2 MultiDiscrete Action Space

                    if self.p2Brain == None:
                        movActP2 = action[1]
                        attActP2 = action[2]
                    else:
                        if self.p2Brain.id == "rl":
                            self.lastObs[:,:,-1] = P2ToP1AddObsMove(self.lastObs[:,:,-1])
                        [movActP2, attActP2], _ = self.p2Brain.act(self.lastObs)

                else: # P2 Discrete Action Space

                    if self.p2Brain == None:
                        movActP2, attActP2 = self.discreteToMultiDiscreteAction(action[1])
                    else:
                        if self.p2Brain.id == "rl":
                            self.lastObs[:,:,-1] = P2ToP1AddObsMove(self.lastObs[:,:,-1])
                        brainActions, _ = self.p2Brain.act(self.lastObs)
                        movActP2, attActP2 = self.discreteToMultiDiscreteAction(brainActions)

        if self.playerSide == "P1P2":
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
        if self.playerSide == "P1P2":
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
               oldRew = info["rewards"]
               observation, info = self.env.continue_game()
               info["rewards"] = oldRew
               self.playingCharacters = self.env.playingCharacters
               self.playerSide = self.env.player
               self.playerId = self.env.playerId
            else:
               print("Episode done")
               done = True
        elif stage_done:
            print("Stage done")
            self.clearActBuf()
            oldRew = info["rewards"]
            observation, info = self.env.next_stage()
            info["rewards"] = oldRew
        elif round_done:
            print("Round done")
            self.clearActBuf()
            oldRew = info["rewards"]
            observation, info = self.env.next_round()
            info["rewards"] = oldRew

        return observation, reward, done, info

    # Resetting the environment
    def reset(self):

        self.clearActBuf()
        self.ncontinue = 0

        if self.first:
            self.first = False
            observation, info = self.env.start(gamePads=self.gamePads)
        else:
            observation, info = self.env.new_game()

        self.playingCharacters = self.env.playingCharacters
        self.playerSide = self.env.player
        self.playerId = self.env.playerId

        # Deactivating showFinal for 2P Env (Needed to do it here after Env start)
        if self.playerSide == "P1P2":
            self.showFinal = False

        return observation

    # Rendering the environment
    def render(self, mode='human'):
        pass

    # Closing the environment
    def close (self):
        self.first = True
        self.env.close()
