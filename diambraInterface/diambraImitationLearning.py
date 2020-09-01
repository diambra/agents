import sys, platform, os
import numpy as np
import gym
from gym import spaces
import pickle, bz2

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
                                        shape=(self.obsH, self.obsW, self.obsNChannels), dtype=np.uint8)

        # List of RL trajectories files
        self.trajFilesList = trajFilesList

        # CPU rank for this env instance
        self.rank = rank
        self.totalCpus = totalCpus

        # Idx of trajectory file to read
        self.trajIdx = self.rank
        self.RLTrajDict = None

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

        # Reward retrieval
        reward = self.RLTrajDict["rewards"][self.stepIdx]

        # Done retrieval
        done = False
        if self.stepIdx == self.RLTrajDict["epLen"] - 1:
            done = True

        # Action retrieval
        action = self.RLTrajDict["actions"][self.stepIdx]

        if np.any(done):
            print("Episode done")

        # Update step idx
        self.stepIdx += 1

        return observation, reward, done, action

    # Resetting the environment
    def reset(self):

        # Reset run step
        self.stepIdx = 0

        # Check if run out of traj files
        if self.trajIdx >= len(self.trajFilesList):
            raise "Exceeded number of RL Traj files"

        RLTrajFile = self.trajFilesList[self.trajIdx]
        # Move traj idx to the next to be read
        self.trajIdx += self.totalCpus

        # Read compressed RL Traj file
        infile = bz2.BZ2File(RLTrajFile, 'r')
        self.RLTrajDict = pickle.load(infile)
        infile.close()

        # Storing env info
        self.numberOfCharacters = self.RLTrajDict["nChars"]
        self.actBufLen = self.RLTrajDict["actBufLen"]

        # Reset observation retrieval
        observation = np.zeros((self.obsH, self.obsW, self.obsNChannels))
        for iFrame in range(self.obsNChannels-1):
            observation[:,:,iFrame] = self.RLTrajDict["frames"][iFrame]
        observation[:,:,self.obsNChannels-1] = self.RLTrajDict["addObs"][0]

        return observation

    # Rendering the environment
    def render(self, mode='human'):
        pass

# Function to vectorialize envs
def make_diambra_imitationLearning_env(diambraIL, diambraIL_kwargs, seed=0):
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
            env = Monitor(env, log_dir)
            return env
        set_global_seeds(seed)
        return _thunk

    return SubprocVecEnv([make_env(i) for i in range(num_env)])
