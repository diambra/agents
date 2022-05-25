import sys, os
import diambraArena
from wrappers.addObsWrap import AdditionalObsToChannel
from wrappers.p2Wrap import selfPlayVsRL, vsHum, integratedSelfPlay

from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

def makeStableBaselinesEnv(seed, envSettings, wrappersSettings=None,
                           trajRecSettings=None, customWrappers=None, keyToAdd=None,
                           p2Mode=None, p2Policy=None, startIndex=0, allowEarlyResets=True,
                           startMethod=None, noVec=False, useSubprocess=False):
    """
    Create a wrapped, monitored VecEnv.
    :param seed: (int) initial seed for RNG
    :param envSettings: (dict) parameters for DIAMBRA environment
    :param wrappersSettings: (dict) parameters for environment wraping function
    :param trajRecSettings: (dict) parameters for environment recording wraping function
    :param keyToAdd: (list) ordered parameters for environment stable baselines converter wraping function
    :param startIndex: (int) start rank index
    :param allowEarlyResets: (bool) allows early reset of the environment
    :param startMethod: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param useSubprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
    :param noVec: (bool) Whether to avoid usage of Vectorized Env or not. Default: False
    :return: (VecEnv) The diambra environment
    """

    envAddresses = os.getenv("DIAMBRA_ENVS", "").split()
    if len(envAddresses) == 0:
        raise Exception("No environments found, use diambra to run your training scripts")

    numEnvs = len(envAddresses)

    hardCore = False
    if "hardCore" in envSettings:
        hardCore = envSettings["hardCore"]

    def makeSbEnv(rank):
        def thunk():
            env = diambraArena.make(envSettings["gameId"], envSettings, wrappersSettings,
                                    trajRecSettings, seed=seed+rank, rank=rank)
            if not hardCore:

                # Applying custom wrappers
                if customWrappers != None:
                    for wrap in customWrappers:
                        env = wrap(env)

                env = AdditionalObsToChannel(env, keyToAdd)
            if p2Mode != None:
                if p2Mode == "integratedSelfPlay":
                    env = integratedSelfPlay(env)
                elif p2Mode == "selfPlayVsRL":
                    env = selfPlayVsRL(env, p2Policy)
                elif p2Mode == "vsHum":
                    env = vsHum(env, p2Policy)

            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allowEarlyResets)
            return env
        return thunk
    set_global_seeds(seed)

    # If not wanting vectorized envs
    if noVec and numEnvs == 1:
        return makeSbEnv(0)(), numEnvs

    # When using one environment, no need to start subprocesses
    if numEnvs == 1 or not useSubprocess:
        return DummyVecEnv([makeSbEnv(i + startIndex) for i in range(numEnvs)]), numEnvs

    return SubprocVecEnv([makeSbEnv(i + startIndex) for i in range(numEnvs)],
                         start_method=startMethod), numEnvs
