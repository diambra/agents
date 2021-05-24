import sys, os, time, random
import numpy as np
from collections import deque
import cv2  # pytype:disable=import-error
cv2.ocl.setUseOpenCL(False)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))

import gym
from gym import spaces

import datetime
from parallelPickle import parallelPickleWriter

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noOpMax=6):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be first action (0).
        :param env: (Gym Environment) the environment to wrap
        :param noOpMax: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.noOpMax = noOpMax
        self.overrideNumNoOps = None

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.overrideNumNoOps is not None:
            noOps = self.overrideNumNoOps
        else:
            noOps = random.randint(1, self.noOpMax + 1)
        assert noOps > 0
        obs = None
        noopAction = [0, 0, 0, 0]
        if (self.env.actionSpace[0] == "discrete") and (self.env.playerSide != "P1P2" or\
                                                        self.env.p2Brain != None):
            noopAction = 0
        for _ in range(noOps):
            obs, _, done, _ = self.env.step(noopAction)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame (frameskipping)
        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)

        # most recent raw observations (for max pooling across time steps)
        self.obsBuffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self.skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        totalReward = 0.0
        done = None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            if i == self.skip - 2:
                self.obsBuffer[0] = obs
            if i == self.skip - 1:
                self.obsBuffer[1] = obs
            totalReward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        maxFrame = self.obsBuffer.max(axis=0)

        return maxFrame, totalReward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        """
        clips the reward to {+1, 0, -1} by its sign.
        :param env: (Gym Environment) the environment
        """
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """
        Bin reward to {+1, 0, -1} by its sign.
        :param reward: (float)
        """
        return np.sign(reward)

class NormalizeRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        """
        Normalize the reward dividing by the 50% of the maximum character health.
        :param env: (Gym Environment) the environment
        """
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """
        Nomralize reward dividing by reward normalization factor*maxHealth.
        :param reward: (float)
        """
        return float(reward)/float(self.env.rewNormFac*self.env.maxHealth)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, hwObsResize = [84, 84]):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = hwObsResize[1]
        self.height = hwObsResize[0]
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class WarpFrame3C(gym.ObservationWrapper):
    def __init__(self, env, hwObsResize = [224, 224]):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = hwObsResize[1]
        self.height = hwObsResize[0]
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3),
                                            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class FrameStack(gym.Wrapper):
    def __init__(self, env, nFrames):
        """Stack nFrames last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames
        :param env: (Gym Environment) the environment
        :param nFrames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.nFrames = nFrames
        self.frames = deque([], maxlen=nFrames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * nFrames),
                                            dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Fill the stack upon reset to avoid black frames
        for _ in range(self.nFrames):
            self.frames.append(obs)
        return self.getOb()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)

        # Add last obs nFrames - 1 times in case of new round / stage / continueGame
        if (info["roundDone"] or info["stageDone"] or info["gameDone"]) and not done:
            for _ in range(self.nFrames - 1):
                self.frames.append(obs)

        return self.getOb(), reward, done, info

    def getOb(self):
        assert len(self.frames) == self.nFrames
        return LazyFrames(list(self.frames))


class FrameStackDilated(gym.Wrapper):
    def __init__(self, env, nFrames, dilation):
        """Stack nFrames last frames with dilation factor.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames
        :param env: (Gym Environment) the environment
        :param nFrames: (int) the number of frames to stack
        :param dilation: (int) the dilation factor
        """
        gym.Wrapper.__init__(self, env)
        self.nFrames = nFrames
        self.dilation = dilation
        self.frames = deque([], maxlen=nFrames*dilation) # Keeping all nFrames*dilation in memory,
                                                          # then extract the subset given by the dilation factor
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * nFrames),
                                            dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.nFrames*self.dilation):
            self.frames.append(obs)
        return self.getOb()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)

        # Add last obs nFrames - 1 times in case of new round / stage / continueGame
        if (info["roundDone"] or info["stageDone"] or info["gameDone"]) and not done:
            for _ in range(self.nFrames*self.dilation - 1):
                self.frames.append(obs)

        return self.getOb(), reward, done, info

    def getOb(self):
        framesSubset = list(self.frames)[self.dilation-1::self.dilation]
        assert len(framesSubset) == self.nFrames
        return LazyFrames(list(framesSubset))


class ScaledFloatFrameNeg(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return (np.array(observation).astype(np.float32) / 127.5) - 1.0

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to np.ndarray before being passed to the model.
        :param frames: ([int] or [float]) environment frames
        """
        self.frames = frames
        self.out = None

    def force(self):
        if self.out is None:
            self.out = np.concatenate(self.frames, axis=2)
            self.frames = None
        return self.out

    def __array__(self, dtype=None):
        out = self.force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.force())

    def __getitem__(self, i):
        return self.force()[i]


def makeDiambra(diambraGame, envId, diambraKwargs, diambraGymKwargs):
    """
    Create a wrapped diambra Environment
    :param envId: (str) the environment ID
    :return: (Gym Environment) the wrapped diambra environment
    """

    env = diambraGame(envId, diambraKwargs, **diambraGymKwargs)
    env = NoopResetEnv(env, noOpMax=6)
    #env = MaxAndSkipEnv(env, skip=4)
    return env

# Deepmind env processing (rewards normalization, resizing, grayscaling, etc)
def wrapDeepmind(env, clipRewards=True, normalizeRewards=False, frameStack=1,
                 scale=False, scaleMod = 0, hwcObsResize = [84, 84, 1], dilation=1):
    """
    Configure environment for DeepMind-style Atari.
    :param env: (Gym Environment) the diambra environment
    :param clipRewards: (bool) wrap the reward clipping wrapper
    :param normalizeRewards: (bool) wrap the reward normalizing wrapper
    :param frameStack: (int) wrap the frame stacking wrapper using #frameStack frames
    :param dilation (frame stacking): (int) stack one frame every #dilation frames, useful
                                            to assure action every step considering a dilated
                                            subset of previous frames
    :param scale: (bool) wrap the scaling observation wrapper
    :return: (Gym Environment) the wrapped diambra environment
    """

    if hwcObsResize[2] == 1:
       # Resizing observation from H x W x 3 to hwObsResize[0] x hwObsResize[1] x 1
       env = WarpFrame(env, hwcObsResize)
    elif hwcObsResize[2] == 3:
       # Resizing observation from H x W x 3 to hwObsResize[0] x hwObsResize[1] x hwObsResize[2]
       env = WarpFrame3C(env, hwcObsResize)
    else:
       raise ValueError("Number of channel must be either 3 or 1")

    # Normalize rewards
    if normalizeRewards:
       env = NormalizeRewardEnv(env)

    # Clip rewards using sign function
    if clipRewards:
        env = ClipRewardEnv(env)

    # Stack #frameStack frames together
    if frameStack > 1:
        if dilation == 1:
            env = FrameStack(env, frameStack)
        else:
            print("Using frame stacking with dilation = {}".format(dilation))
            env = FrameStackDilated(env, frameStack, dilation)

    # Scales observations normalizing them
    if scale:
        if scaleMod == 0:
           # Between 0.0 and 1.0
           env = ScaledFloatFrame(env)
        elif scaleMod == -1:
           # Between -1.0 and 1.0
           env = ScaledFloatFrameNeg(env)
        else:
           raise ValueError("Scale mod musto be either 0 or -1")

    return env

# Diambra additional observations (previous moves, character side, ecc)
class AddObs(gym.Wrapper):
    def __init__(self, env, keyToAdd):
        """
        Add to observations additional info requested via `keyToAdd` str list
        :param env: (Gym Environment) the environment to wrap
        :param keyToAdd: (list of str) list of info to add to the observation
        """
        gym.Wrapper.__init__(self, env)
        self.keyToAdd = keyToAdd
        shp = self.env.observation_space.shape

        self.boxHighBound = self.env.observation_space.high.max()
        self.boxLowBound = self.env.observation_space.low.min()
        assert (self.boxHighBound == 1.0 or self.boxHighBound == 255),\
                "Observation space max bound must be either 1.0 or 255 to use Additional Obs"
        assert (self.boxLowBound == 0.0 or self.boxLowBound == -1.0),\
                "Observation space min bound must be either 0.0 or -1.0 to use Additional Obs"

        self.observation_space = spaces.Box(low=self.boxLowBound, high=self.boxHighBound,
                                            shape=(shp[0], shp[1], shp[2] + 1),
                                            dtype=np.float32)

        # Initialize last observation
        self.env.updateLastObs(np.zeros((shp[0], shp[1], shp[2]+1), dtype=self.env.observation_space.dtype))

        self.resetInfo = {}
        self.resetInfo["actionsBufP1"] = np.concatenate(
                                           (self.actionsVector([0 for i in range(self.env.actBufLen)],
                                                               self.env.nActions[0][0]),
                                            self.actionsVector([0 for i in range(self.env.actBufLen)],
                                                               self.env.nActions[0][1]))
                                                      )
        self.resetInfo["actionsBufP2"] = np.concatenate(
                                           (self.actionsVector([0 for i in range(self.env.actBufLen)],
                                                               self.env.nActions[1][0]),
                                            self.actionsVector([0 for i in range(self.env.actBufLen)],
                                                               self.env.nActions[1][1]))
                                                      )

        if "ownHealth" in self.keyToAdd:
            self.resetInfo["ownHealthP1"] = [1]
            self.resetInfo["oppHealthP1"] = [1]
            self.resetInfo["ownHealthP2"] = [1]
            self.resetInfo["oppHealthP2"] = [1]
        else:
            self.resetInfo["ownHealth_1P1"] = [1]
            self.resetInfo["ownHealth_2P1"] = [1]
            self.resetInfo["oppHealth_1P1"] = [1]
            self.resetInfo["oppHealth_2P1"] = [1]
            self.resetInfo["ownActiveCharP1"] = [0]
            self.resetInfo["oppActiveCharP1"] = [0]

            self.resetInfo["ownHealth_1P2"] = [1]
            self.resetInfo["ownHealth_2P2"] = [1]
            self.resetInfo["oppHealth_1P2"] = [1]
            self.resetInfo["oppHealth_2P2"] = [1]
            self.resetInfo["ownActiveCharP2"] = [0]
            self.resetInfo["oppActiveCharP2"] = [0]
        if self.env.playerSide == "P1" or self.env.playerSide == "P1P2":
            self.resetInfo["ownPositionP1"] = [0]
            self.resetInfo["oppPositionP1"] = [1]
        else:
            self.resetInfo["ownPositionP1"] = [1]
            self.resetInfo["oppPositionP1"] = [0]
        self.resetInfo["ownPositionP2"] = [1]
        self.resetInfo["oppPositionP2"] = [0]
        self.resetInfo["ownWinsP1"] = [0]
        self.resetInfo["oppWinsP1"] = [0]
        self.resetInfo["ownWinsP2"] = [0]
        self.resetInfo["oppWinsP2"] = [0]
        self.resetInfo["stageP1"] = [0.0]
        self.resetInfo["stageP2"] = [0.0]

    # Update playing char
    def updatePlayingChar(self, dictToUpdate):

        tmpChar1 = np.zeros(self.env.numberOfCharacters)
        tmpChar2 = np.zeros(self.env.numberOfCharacters)
        if self.env.playerSide != "P1P2":
            tmpChar1[self.env.playingCharacters[self.env.playerId]] = 1
        else :
            tmpChar1[self.env.playingCharacters[0]] = 1
            tmpChar2[self.env.playingCharacters[1]] = 1
        dictToUpdate["characterP1"] = tmpChar1
        dictToUpdate["characterP2"] = tmpChar2

        return

    # Building the one hot encoding actions vector
    def actionsVector(self, actionsBuf, nAct):

        actionsVector = np.zeros( (len(actionsBuf), nAct), dtype=int)

        for iAction, _ in enumerate(actionsBuf):
           actionsVector[iAction][actionsBuf[iAction]] = 1

        actionsVector = np.reshape(actionsVector, [-1])

        return actionsVector

    # Observation modification (adding one channel to store additional info)
    def observationMod(self, obs, additionalInfo):

        shp = self.observation_space.shape

        # Adding a channel to the standard image, it will be in last position and it will store additional obs
        obsNew = np.zeros((shp[0], shp[1], shp[2]), dtype=self.env.observation_space.dtype)

        # Storing standard image in the first channel leaving the last one for additional obs
        obsNew[:,:,0:shp[2]-1] = obs

        # Creating the additional channel where to store new info
        obsNewAdd = np.zeros((shp[0], shp[1], 1), dtype=self.env.observation_space.dtype)

        # Adding new info to the additional channel, on a very
        # long line and then reshaping into the obs dim
        newData = np.zeros((shp[0] * shp[1]))

        # Adding new info for 1P
        counter = 0
        for key in self.keyToAdd:

            for addInfo in additionalInfo[key+"P1"]:

                counter = counter + 1
                newData[counter] = addInfo

        newData[0] = counter

        # Adding new info for P2 in 2P games
        if self.env.playerSide == "P1P2":
            halfPosIdx = int((shp[0] * shp[1]) / 2)
            counter = halfPosIdx

            for key in self.keyToAdd:

                for addInfo in additionalInfo[key+"P2"]:

                    counter = counter + 1
                    newData[counter] = addInfo

            newData[halfPosIdx] = counter - halfPosIdx

        newData = np.reshape(newData, (shp[0], -1))

        newData = newData * self.boxHighBound

        obsNew[:,:,shp[2]-1] = newData

        return obsNew

    # Creating dictionary for additional info of the step
    def toStepInfo(self, info, action):

        stepInfo = {}
        stepInfo["actionsBufP1"] = np.concatenate(
                                      (self.actionsVector( info["actionsBufP1"][0], self.env.nActions[0][0] ),
                                       self.actionsVector( info["actionsBufP1"][1], self.env.nActions[0][1] ))
                                                  )
        if self.env.playerSide == "P1P2":
            stepInfo["actionsBufP2"] = np.concatenate(
                                          (self.actionsVector( info["actionsBufP2"][0], self.env.nActions[1][0] ),
                                           self.actionsVector( info["actionsBufP2"][1], self.env.nActions[1][1] ))
                                                      )

        if self.env.playerSide == "P1" or self.env.playerSide == "P1P2":

            if "ownHealth" in self.keyToAdd:
                stepInfo["ownHealthP1"] = [info["healthP1"] / float(self.env.maxHealth)]
                stepInfo["oppHealthP1"] = [info["healthP2"] / float(self.env.maxHealth)]
            else:
                stepInfo["ownHealth_1P1"] = [info["healthP1_1"] / float(self.env.maxHealth)]
                stepInfo["ownHealth_2P1"] = [info["healthP1_2"] / float(self.env.maxHealth)]
                stepInfo["oppHealth_1P1"] = [info["healthP2_1"] / float(self.env.maxHealth)]
                stepInfo["oppHealth_2P1"] = [info["healthP2_2"] / float(self.env.maxHealth)]

                stepInfo["ownActiveCharP1"] = [info["activeCharP1"]]
                stepInfo["oppActiveCharP1"] = [info["activeCharP2"]]

            stepInfo["ownPositionP1"] = [info["positionP1"]]
            stepInfo["oppPositionP1"] = [info["positionP2"]]

            stepInfo["ownWinsP1"] = [info["winsP1"]]
            stepInfo["oppWinsP1"] = [info["winsP2"]]
        else:
            if "ownHealth" in self.keyToAdd:
                stepInfo["ownHealthP1"] = [info["healthP2"] / float(self.env.maxHealth)]
                stepInfo["oppHealthP1"] = [info["healthP1"] / float(self.env.maxHealth)]
            else:
                stepInfo["ownHealth_1P1"] = [info["healthP2_1"] / float(self.env.maxHealth)]
                stepInfo["ownHealth_2P1"] = [info["healthP2_2"] / float(self.env.maxHealth)]
                stepInfo["oppHealth_1P1"] = [info["healthP1_1"] / float(self.env.maxHealth)]
                stepInfo["oppHealth_2P1"] = [info["healthP1_2"] / float(self.env.maxHealth)]

                stepInfo["ownActiveCharP1"] = [info["activeCharP2"]]
                stepInfo["oppActiveCharP1"] = [info["activeCharP1"]]

            stepInfo["ownPositionP1"] = [info["positionP2"]]
            stepInfo["oppPositionP1"] = [info["positionP1"]]

            stepInfo["ownWinsP1"] = [info["winsP2"]]
            stepInfo["oppWinsP1"] = [info["winsP1"]]

        stepInfo["stageP1"] = [ float(info["stage"]-1) / float(self.env.maxStage - 1) ]

        if self.env.playerSide == "P1P2":
            if "ownHealth" in self.keyToAdd:
                stepInfo["ownHealthP2"] = stepInfo["oppHealthP1"]
                stepInfo["oppHealthP2"] = stepInfo["ownHealthP1"]
            else:
                stepInfo["ownHealth_1P2"] = stepInfo["oppHealth_1P1"]
                stepInfo["ownHealth_2P2"] = stepInfo["oppHealth_2P1"]
                stepInfo["oppHealth_1P2"] = stepInfo["ownHealth_1P1"]
                stepInfo["oppHealth_2P2"] = stepInfo["ownHealth_2P1"]

                stepInfo["ownActiveCharP2"] = [info["activeCharP2"]]
                stepInfo["oppActiveCharP2"] = [info["activeCharP1"]]

            stepInfo["ownPositionP2"] = [info["positionP2"]]
            stepInfo["oppPositionP2"] = [info["positionP1"]]

            stepInfo["ownWinsP2"] = [info["winsP2"]]
            stepInfo["oppWinsP2"] = [info["winsP1"]]
            stepInfo["stageP2"] = stepInfo["stageP1"]

        self.updatePlayingChar(stepInfo)

        return stepInfo

    def reset(self, **kwargs):
        """
        Reset the environment and add requested info to the observation
        :return: new observation
        """

        obs = self.env.reset(**kwargs)
        obs = np.array(obs).astype(np.float32)

        self.updatePlayingChar(self.resetInfo)
        obsNew = self.observationMod(obs, self.resetInfo)

        # Store last observation
        self.env.updateLastObs(obsNew)

        return obsNew

    def step(self, action):
        """
        Step the environment with the given action
        and add requested info to the observation
        :param action: ([int] or [float]) the action
        :return: new observation, reward, done, information
        """
        obs, reward, done, info = self.env.step(action)

        stepInfo = self.toStepInfo(info, action)

        obsNew = self.observationMod(obs, stepInfo)

        # Store last observation
        self.env.updateLastObs(obsNew)

        return obsNew, reward, done, info

def additionalObs(env, keyToAdd):
    """
    Add additional observations to the environment output.
    :param env: (Gym Environment) the diambra environment
    :param keyToAdd: (list of str) additional info to add to the obs
    :return: (Gym Environment) the wrapped diambra environment
    """

    if keyToAdd != None:
       env = AddObs(env, keyToAdd)

    return env

# Trajectory recorder wrapper
class TrajectoryRecorder(gym.Wrapper):
    def __init__(self, env, filePath, userName, ignoreP2, commitHash, keyToAdd):
        """
        Record trajectories to use them for imitation learning
        :param env: (Gym Environment) the environment to wrap
        :param filePath: (str) file path specifying where to store the trajectory file
        """
        gym.Wrapper.__init__(self, env)
        self.filePath = filePath
        self.userName = userName
        self.ignoreP2 = ignoreP2
        self.keyToAdd = keyToAdd
        self.shp = self.env.observation_space.shape
        self.commitHash = commitHash

        if self.env.playerSide == "P1P2":
            if ((self.env.attackButCombinations[0] != self.env.attackButCombinations[1]) or\
                (self.env.actionSpace[0] != self.env.actionSpace[1])):
                raise Exception("Different attack buttons combinations and/or "\
                                "different action spaces not supported for 2P experience recordings")

        print("Recording trajectories in \"{}\"".format(self.filePath))
        os.makedirs(self.filePath, exist_ok = True)

    def reset(self, **kwargs):
        """
        Reset the environment and add requested info to the observation
        :return: observation
        """

        # Items to store
        self.lastFrameHist = []
        self.addObsHist = []
        self.rewardsHist = []
        self.actionsHist = []
        self.flagHist = []
        self.cumulativeRew = 0

        obs = self.env.reset(**kwargs)

        for idx in range(self.shp[2]-1):
            self.lastFrameHist.append(obs[:,:,idx])

        self.addObsHist.append(obs[:,:,self.shp[2]-1])

        return obs

    def step(self, action):
        """
        Step the environment with the given action
        and add requested info to the observation
        :param action: ([int] or [float]) the action
        :return: new observation, reward, done, information
        """

        obs, reward, done, info = self.env.step(action)

        self.lastFrameHist.append(obs[:,:,self.shp[2]-2])

        # Add last obs nFrames - 1 times in case of new round / stage / continue_game
        if (info["roundDone"] or info["stageDone"] or info["gameDone"]) and not done:
            for _ in range(self.shp[2]-2):
                self.lastFrameHist.append(obs[:,:,self.shp[2]-2])

        self.addObsHist.append(obs[:,:,self.shp[2]-1])
        self.rewardsHist.append(reward)
        self.actionsHist.append(action)
        self.flagHist.append([info["roundDone"], info["stageDone"],
                              info["gameDone"], info["episodeDone"]])
        self.cumulativeRew += reward

        if done:
            toSave = {}
            toSave["commitHash"]    = self.commitHash
            toSave["userName"]      = self.userName
            toSave["playerId"]      = self.env.playerSide
            toSave["difficulty"]    = self.env.difficulty
            toSave["ignoreP2"]      = self.ignoreP2
            toSave["charNames"]     = self.env.charNames
            toSave["actBufLen"]     = self.env.actBufLen
            toSave["nActions"]      = self.env.nActions[0]
            toSave["attackButComb"] = self.env.attackButCombinations[0]
            toSave["actionSpace"]   = self.env.actionSpace[0]
            toSave["epLen"]         = len(self.rewardsHist)
            toSave["cumRew"]        = self.cumulativeRew
            toSave["keyToAdd"]      = self.keyToAdd
            toSave["frames"]        = self.lastFrameHist
            toSave["addObs"]        = self.addObsHist
            toSave["rewards"]       = self.rewardsHist
            toSave["actions"]       = self.actionsHist
            toSave["doneFlags"]     = self.flagHist

            # Characters name
            chars = ""
            # If 2P mode
            if self.env.playerSide == "P1P2" and self.ignoreP2 == 0:
                chars += self.env.charNames[self.env.playingCharacters[0]]
                chars += self.env.charNames[self.env.playingCharacters[1]]
            # If 1P mode
            else:
                chars += self.env.charNames[self.env.playingCharacters[self.env.playerId]]

            savePath = "mod" + str(self.ignoreP2) + "_" + self.env.playerSide + "_" + chars +\
                       "_diff" + str(self.env.difficulty)  + "_rew" + str(np.round(self.cumulativeRew, 3)) +\
                       "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            pickleWriter = parallelPickleWriter(os.path.join(self.filePath, savePath), toSave)
            pickleWriter.start()

        return obs, reward, done, info
