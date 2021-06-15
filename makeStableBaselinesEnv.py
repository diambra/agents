from makeEnv import *

from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

def makeStableBaselinesEnv(envPrefix, numEnv, seed, diambraKwargs, diambraGymKwargs,
                           wrapperKwargs=None, trajRecKwargs=None, hardCore=False,
                           startIndex=0, allowEarlyResets=True, startMethod=None,
                           keyToAdd=None, noVec=False, useSubprocess=False):
    """
    Create a wrapped, monitored VecEnv for Atari.
    :param numEnv: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapperKwargs: (dict) the parameters for wrapDeepmind function
    :param startIndex: (int) start rank index
    :param allowEarlyResets: (bool) allows early reset of the environment
    :param startMethod: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param useSubprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
    :param noVec: (bool) Whether to avoid usage of Vectorized Env or not. Default: False
    :return: (VecEnv) The diambra environment
    """

    def makeSbEnv(rank):
        def thunk():
            envId = envPrefix + str(rank)
            env = makeEnv(envId, seed+rank, diambraKwargs, diambraGymKwargs,
                          wrapperKwargs, trajRecKwargs, hardCore)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allowEarlyResets)
            return env
        return thunk
    set_global_seeds(seed)

    # If not wanting vectorized envs
    if noVec and numEnv == 1:
        envId = envPrefix + str(0)
        env = makeEnv(envId, seed, diambraKwargs, diambraGymKwargs, wrapperKwargs,
                      trajRecKwargs, hardCore)
        env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                      allow_early_resets=allowEarlyResets)
        return env

    # When using one environment, no need to start subprocesses
    if numEnv == 1 or not useSubprocess:
        return DummyVecEnv([makeSbEnv(i + startIndex) for i in range(numEnv)])

    return SubprocVecEnv([makeSbEnv(i + startIndex) for i in range(numEnv)],
                         start_method=startMethod)
