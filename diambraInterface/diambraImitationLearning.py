import sys, platform, os
import numpy as np
import gym
from gym import spaces
import pickle, bz2
import copy
import cv2

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor

class diambraImitationLearning(gym.Env):
    """DiambraMame Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, hwc_dim, n_actions, trajFilesList, rank, totalCpus):
        super(diambraImitationLearning, self).__init__()

        # Observation and action space
        self.obsH = hwc_dim[0]
        self.obsW = hwc_dim[1]
        self.obsNChannels = hwc_dim[2]
        self.n_actions = n_actions

        # Define action and observation space
        # They must be gym.spaces objects
        # MultiDiscrete actions:
        # - Arrows -> One discrete set
        # - Buttons -> One discrete set
        # NB: use the convention NOOP = 0, and buttons combinations are prescripted,
        #     e.g. NOOP = [0], ButA = [1], ButB = [2], ButA+ButB = [3]
        self.action_space = spaces.MultiDiscrete(self.n_actions)

        # Image as input:
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                        shape=(self.obsH, self.obsW, self.obsNChannels), dtype=np.float32)

        # List of RL trajectories files
        self.trajFilesList = trajFilesList

        # CPU rank for this env instance
        self.rank = rank
        self.totalCpus = totalCpus

        # Idx of trajectory file to read
        self.trajIdx = self.rank
        self.RLTrajDict = None

        # If run out of examples
        self.exhausted = False

        # Reset flag
        self.nReset = 0

    # Print Episode summary
    def trajSummary(self):

        print(self.RLTrajDict.keys())

        print("Ep. length =", self.RLTrajDict["epLen"] )

        for key, value in self.RLTrajDict.items():
            if type(value) == list:
                print("len(",key,") :", len(value))
            else:
                print(key,":", value)

    # Step the environment
    def step(self, dummyAction):

        # Observation retrieval
        observation = np.zeros((self.obsH, self.obsW, self.obsNChannels))
        for iFrame in range(self.obsNChannels-1):
            observation[:,:,iFrame] = self.RLTrajDict["frames"][self.stepIdx + 1 + iFrame]
        observation[:,:,self.obsNChannels-1] = self.RLTrajDict["addObs"][self.stepIdx + 1]

        # Storing last observation for rendering
        self.lastObs = observation[:,:,self.obsNChannels-2]

        # Reward retrieval
        reward = self.RLTrajDict["rewards"][self.stepIdx]

        # Done retrieval
        done = False
        if self.stepIdx == self.RLTrajDict["epLen"] - 1:
            done = True

        # Action retrieval
        action = self.RLTrajDict["actions"][self.stepIdx]
        action = [action[0], action[1]]
        info = {}
        info["action"] = action

        if np.any(done):
            print("(Rank", self.rank, ") Episode done")

        # Update step idx
        self.stepIdx += 1

        return observation, reward, done, info

    # Resetting the environment
    def reset(self):

        # Reset run step
        self.stepIdx = 0

        # Check if run out of traj files
        if self.trajIdx >= len(self.trajFilesList):
            print("(Rank", self.rank, ") Resetting env")
            self.exhausted = True
            return [None]

        if self.nReset == 0:
            RLTrajFile = self.trajFilesList[self.trajIdx]

            # Read compressed RL Traj file
            infile = bz2.BZ2File(RLTrajFile, 'r')
            self.RLTrajDict = pickle.load(infile)
            infile.close()

            # Storing env info
            self.nChars = self.RLTrajDict["nChars"]
            self.actBufLen = self.RLTrajDict["actBufLen"]
            self.playerId = self.RLTrajDict["playerId"]

        if self.playerId == "P1P2":

            print("Two players RL trajectory")

            if self.nReset == 0:
            # First reset for this trajectory

                print("Loading P1 data for 2P trajectory")

                # Generate P2 Experience from P1 one
                self.generateP2ExperienceFromP1()

                # Correct additional observation for P1
                self.RLTrajDict["addObs"] = self.AddObsCorrection(self.RLTrajDict["addObs"], player_id=0)

                # Update reset counter
                self.nReset += 1

            else:
            # Second reset for this trajectory

                print("Loading P2 data for 2P trajectory")

                # OverWrite P1 RL trajectory with the one calculated for P2
                self.RLTrajDict = self.RLTrajDictP2

                # Reset reset counter
                self.nReset = 0

                # Move traj idx to the next to be read
                self.trajIdx += self.totalCpus

        else:

            print("One player RL trajectory")

            # Move traj idx to the next to be read
            self.trajIdx += self.totalCpus

        # Reset observation retrieval
        observation = np.zeros((self.obsH, self.obsW, self.obsNChannels))
        for iFrame in range(self.obsNChannels-1):
            observation[:,:,iFrame] = self.RLTrajDict["frames"][iFrame]
        observation[:,:,self.obsNChannels-1] = self.RLTrajDict["addObs"][0]

        # Storing last observation for rendering
        self.lastObs = observation[:,:,self.obsNChannels-2]

        return observation

    # Correct additional info from P1P2 mode to 1P (for both P1 and P2)
    # It rebuilds additional Obs as if it had been generated by 1P game (only 1P actions)
    def AddObsCorrection(self, addObs, player_id):

        newAddObsList = []
        # For each step, correct additional observation
        for observation in addObs:

            newAddObs = np.reshape(np.zeros(addObs[0].shape), (-1))

            additionalPar = int(observation[0,0])
            nScalarAddPar = additionalPar - 2*self.nChars - 2*self.actBufLen*(self.n_actions[0]+self.n_actions[1])
            addPar = observation[:,:]
            addPar = np.reshape(addPar, (-1))[1:additionalPar+1]
            actions = addPar[0:additionalPar-nScalarAddPar-2*self.nChars]

            limAct = [self.actBufLen * self.n_actions[0],
                      self.actBufLen * self.n_actions[0] + self.actBufLen * self.n_actions[1]]

            moveActionsP1   = actions[0:limAct[0]]
            attackActionsP1 = actions[limAct[0]:limAct[1]]

            moveActionsP2   = actions[limAct[1]:limAct[1]+limAct[0]]
            attackActionsP2 = actions[limAct[1]+limAct[0]:limAct[1]+limAct[1]]

            others = addPar[additionalPar-nScalarAddPar-2*self.nChars:]

            P1Health = others[0]
            P2Health = others[1]
            P1Position = others[2]
            P2Position = others[3]
            P1Char = others[nScalarAddPar              : nScalarAddPar +     self.nChars]
            P2Char = others[nScalarAddPar + self.nChars: nScalarAddPar + 2 * self.nChars]

            newAdditionalPar = len(moveActionsP1) + len(attackActionsP1) + nScalarAddPar + 2*self.nChars
            newAddObs[0] = float(newAdditionalPar)

            if player_id == 0:
                newAddObs[1:1+newAdditionalPar] = np.append( np.append( np.append( np.append(moveActionsP1, attackActionsP1),
                                                                                   [P1Health, P2Health, P1Position, P2Position]),
                                                                        P1Char),
                                                             P2Char)
            else:
                newAddObs[1:1+newAdditionalPar] = np.append( np.append( np.append( np.append(moveActionsP2, attackActionsP2),
                                                                                   [P2Health, P1Health, P2Position, P1Position]),
                                                                        P2Char),
                                                             P1Char)

            newAddObs = np.reshape(newAddObs, (addObs[0].shape[0], -1))
            newAddObsList.append(newAddObs)

        return newAddObsList

    # Generate P2 Experience from P1 one
    def generateP2ExperienceFromP1(self):

        # Copy P1 Trajectory
        self.RLTrajDictP2 = copy.deepcopy(self.RLTrajDict)

        # Additional Observations
        self.RLTrajDictP2["addObs"] = self.AddObsCorrection(self.RLTrajDict["addObs"], player_id=1)

        # For each step, convert P1 into P2 experience
        for idx in range(self.RLTrajDict["epLen"]):

            # Rewards (inverting sign)
            self.RLTrajDictP2["rewards"][idx] = -self.RLTrajDict["rewards"][idx]

            # Actions (inverting positions)
            self.RLTrajDictP2["actions"][idx] = [self.RLTrajDict["actions"][idx][2],
                                                 self.RLTrajDict["actions"][idx][3]]

    # Rendering the environment
    def render(self, mode='human'):

        if mode == "human":
            windowName = "Diambra Imitation Learning Environment"
            cv2.namedWindow(windowName,cv2.WINDOW_GUI_NORMAL)
            cv2.imshow(windowName, self.lastObs)
            cv2.waitKey(1)
        elif mode == "rgb_array":
            output = np.expand_dims(self.lastObs, axis=2)
            return output

# Function to vectorialize envs
def make_diambra_imitationLearning_env(diambraIL, diambraIL_kwargs, seed=0,
                                       allow_early_resets=True):
    """
    Utility function for multiprocessed env.

    :param diambraIL_kwargs: (dict) kwargs for Diambra IL env
    """

    num_env = diambraIL_kwargs["totalCpus"]

    def make_env(rank):
        def _thunk():

            # Create log dir
            log_dir = "tmp"+str(rank)+"/"
            os.makedirs(log_dir, exist_ok=True)
            env = diambraIL(**diambraIL_kwargs, rank=rank)
            env = Monitor(env, log_dir, allow_early_resets=allow_early_resets)
            return env
        set_global_seeds(seed)
        return _thunk

    # When using one environment, no need to start subprocesses
    if num_env == 1:
        return DummyVecEnv([make_env(i) for i in range(num_env)])

    return SubprocVecEnv([make_env(i) for i in range(num_env)])
