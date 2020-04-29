import random
import numpy as np
from collections import deque
import cv2  # pytype:disable=import-error
cv2.ocl.setUseOpenCL(False)

import gym
from gym import spaces

from diambraMameGym import *

from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=6):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be last action (env.action_space.n - 1).
        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = env.action_space.n - 1
        print("Noop action N = ", self.noop_action)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
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
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

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
        Nomralize reward dividing by 0.5*max_health.
        :param reward: (float)
        """
        return float(reward)/float(0.5*self.max_health)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, hw_obs_resize = [84, 84]):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = hw_obs_resize[1]
        self.height = hw_obs_resize[0]
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


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames
        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                            dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return LazyFrames(list(self.frames))


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
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


def make_diambra(diambraGame, env_id, diambra_kwargs, continue_game):
    """
    Create a wrapped diambra Environment
    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped diambra environment
    """

    env = diambraGame(env_id, diambra_kwargs, continue_game)
    env = NoopResetEnv(env, noop_max=6)
    #env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, clip_rewards=True, normalize_rewards=False, frame_stack=1, scale=False, hw_obs_resize = [84, 84]):
    """
    Configure environment for DeepMind-style Atari.
    :param env: (Gym Environment) the diambra environment
    :param clip_rewards: (bool) wrap the reward clipping wrapper
    :param normalize_rewards: (bool) wrap the reward normalizing wrapper
    :param frame_stack: (int) wrap the frame stacking wrapper using #frame_stack frames
    :param scale: (bool) wrap the scaling observation wrapper
    :return: (Gym Environment) the wrapped diambra environment
    """

    # Resizing observation from H x W x 3 to hw_obs_resize[0] x hw_obs_resize[1] x 1
    env = WarpFrame(env, hw_obs_resize)

    # Normalize rewards
    if normalize_rewards:
       env = NormalizeRewardEnv(env)

    # Clip rewards using sign function
    if clip_rewards:
        env = ClipRewardEnv(env)

    # Stack #frame_stack frames together
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)

    # Scales observations normalizing them between 0.0 and 1.0
    if scale:
        env = ScaledFloatFrame(env)

    return env

class AddObs(gym.Wrapper):
    def __init__(self, env, key_to_add):
        """
        Add to observations additional info requested via `key_to_add` str list
        :param env: (Gym Environment) the environment to wrap
        :param key_to_add: (list of str) list of info to add to the observation
        """
        gym.Wrapper.__init__(self, env)
        self.key_to_add = key_to_add
        shp = self.env.observation_space.shape

        assert self.env.observation_space.high.max() == 1.0, "Observation space must be normalized [max, min] = [0.0, 1.0] to use Additional Obs"
        assert self.env.observation_space.low.min() == 0.0,  "Observation space must be normalized [max, min] = [0.0, 1.0] to use Additional Obs"
        assert self.env.observation_space.dtype == "float32", "Observation space must have float 32 numbers"

        self.observation_space = spaces.Box(low=0, high=1.0, shape=(shp[0], shp[1], shp[2] + 1),
                                            dtype=env.observation_space.dtype)

        self.playerIdDict = {}
        self.playerIdDict["P1"] = 0
        self.playerIdDict["P2"] = 1

        self.resetInfo = {}
        self.resetInfo["actionsBuf"] = self.actionsVector([self.env.no_op_action for i in range(self.env.actions_buf_len)])
        self.resetInfo["player"] = [self.playerIdDict[self.env.player_id]]
        self.resetInfo["healthP1"] = [1]
        self.resetInfo["healthP2"] = [1]
        self.resetInfo["positionP1"] = [0]
        self.resetInfo["positionP2"] = [1]
        self.resetInfo["winsP1"] = [0]
        self.resetInfo["winsP2"] = [0]

    def actionsVector(self, actionsBuf):

        actionsVector = np.zeros( (len(actionsBuf), self.env.action_space.n ), dtype=int)

        for iAction in range(len(actionsBuf)):
           actionsVector[iAction][actionsBuf[iAction]] = 1

        actionsVector = np.reshape(actionsVector, [-1])

        return actionsVector

    def observation_mod(self, obs, additionalInfo):

        shp = self.observation_space.shape

        # Adding a channel to the standard image, it will be in last position and it will store additional obs
        obsNew = np.zeros((shp[0], shp[1], shp[2]), dtype=self.env.observation_space.dtype)

        # Storing standard image in the first channel leaving the last one for additional obs
        obsNew[:,:,0:shp[2]-1] = obs

        # Creating the additional channel where to store new info
        obsNewAdd = np.zeros((shp[0], shp[1], 1), dtype=self.env.observation_space.dtype)

        # Adding new info to the additional channel, on a very long line and then reshaping into the obs dim
        newData = np.zeros((shp[0] * shp[1]))
        counter = 0
        for key in self.key_to_add:

           for idx in range(len(additionalInfo[key])):

              counter = counter + 1
              newData[counter] = additionalInfo[key][idx]

        newData[0] = counter
        newData = np.reshape(newData, (shp[0], -1))

        obsNew[:,:,shp[2]-1] = newData

        return obsNew

    def to_step_info(self, info):

        step_info = {}
        step_info["actionsBuf"] = self.actionsVector( info["actionsBuf"] )
        step_info["player"] = self.resetInfo["player"]
        step_info["healthP1"] = [info["healthP1"] / float(self.env.max_health)]
        step_info["healthP2"] = [info["healthP2"] / float(self.env.max_health)]
        step_info["positionP1"] = [info["positionP1"]]
        step_info["positionP2"] = [info["positionP2"]]
        step_info["winsP1"] = [info["winsP1"]]
        step_info["winsP2"] = [info["winsP2"]]

        return step_info

    def reset(self, **kwargs):
        """
        Reset the environment and add requested info to the observation
        :param action: ([int] or [float]) the action
        :return: new observation
        """

        obs = self.env.reset(**kwargs)
        obs = np.array(obs).astype(np.float32)

        obsNew = self.observation_mod(obs, self.resetInfo)

        return obsNew

    def step(self, action):
        """
        Step the environment with the given action
        and add requested info to the observation
        :param action: ([int] or [float]) the action
        :return: new observation, reward, done, information
        """
        obs, reward, done, info = self.env.step(action)

        stepInfo = self.to_step_info(info)

        obsNew = self.observation_mod(obs, stepInfo)

        return obsNew, reward, done, info

def additional_obs(env, key_to_add):
    """
    Add additional observations to the environment output.
    :param env: (Gym Environment) the diambra environment
    :param ket_to_add: (list of str) additional info to add to the obs
    :return: (Gym Environment) the wrapped diambra environment
    """

    if key_to_add != None:
       env = AddObs(env, key_to_add)

    return env

def make_diambra_env(diambraMame, env_prefix, num_env, seed, diambra_kwargs, continue_game=1.0, wrapper_kwargs=None,
                   start_index=0, allow_early_resets=True, start_method=None, key_to_add=None,
                   no_vec=False, use_subprocess=False):
    """
    Create a wrapped, monitored VecEnv for Atari.
    :param diambraMame: (class) DIAMBRAGym interface class
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param continue_game: (bool) whether to continue the game after losing
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
    :param no_vec: (bool) Whether to avoid usage of Vectorized Env or not. Default: False
    :return: (VecEnv) The diambra environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        def _thunk():
            env_id = env_prefix + str(rank)
            env = make_diambra(diambraMame, env_id, diambra_kwargs, continue_game)
            env.seed(seed + rank)
            env = wrap_deepmind(env, **wrapper_kwargs)
            env = additional_obs(env, key_to_add)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)
            return env
        return _thunk
    set_global_seeds(seed)

    # If not wanting vectorized envs
    if no_vec and num_env == 1:
        env_id = env_prefix + str(0)
        env = make_diambra(diambraMame, env_id, diambra_kwargs, continue_game)
        env.seed(seed)
        env = wrap_deepmind(env, **wrapper_kwargs)
        env = additional_obs(env, key_to_add)
        env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                      allow_early_resets=allow_early_resets)
        return env

    # When using one environment, no need to start subprocesses
    if num_env == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)],
                         start_method=start_method)
