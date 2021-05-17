from diambraWrappers import *

def makeDiambraEnv(diambraMame, envPrefix, seed, diambraKwargs, diambraGymKwargs,
                     wrapperKwargs=None, trajRecKwargs=None, keyToAdd=None):
    """
    Create a wrapped, monitored VecEnv for Atari.
    :param diambraMame: (class) DIAMBRAGym interface class
    :param seed: (int) the initial seed for RNG
    :param wrapperKwargs: (dict) the parameters for wrapDeepmind function
    """
    if wrapperKwargs is None:
        wrapperKwargs = {}

    env = makeDiambra(diambraMame, envPrefix, diambraKwargs, diambraGymKwargs)
    env.seed(seed)
    env = wrapDeepmind(env, **wrapperKwargs)
    env = additionalObs(env, keyToAdd)
    if type(trajRecKwargs) != type(None):
        env = TrajectoryRecorder(env, **trajRecKwargs, keyToAdd=keyToAdd)

    return env
