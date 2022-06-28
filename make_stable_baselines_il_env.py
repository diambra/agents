import os
import diambraArena
from wrappers.addObsWrap import AdditionalObsToChannel

from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_stable_baselines_il_env(env_prefix, settings, seed, key_to_add=None,
                                 start_index=0, allow_early_resets=True,
                                 start_method=None, no_vec=False,
                                 use_subprocess=False):
    """
    Create a wrapped, monitored VecEnv.
    :param settings: (dict) settings for DIAMBRA IL environment
    :param seed: (int) initial seed for RNG
    :param key_to_add: (list) ordered parameters for environment stable
                       baselines converter wraping function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or
                           `DummyVecEnv` when
    :param no_vec: (bool) Whether to avoid usage of Vectorized Env or not.
                   Default: False
    :return: (VecEnv) The diambra environment
    """

    hardcore = False
    if "hardcore" in settings:
        hardcore = settings["hardcore"]

    def make_sb_env(rank):
        def thunk():
            if hardcore:
                env = ImitationLearningHardcore(**settings, rank=rank)
            else:
                env = ImitationLearning(**settings, rank=rank)
                env = AdditionalObsToChannel(env, key_to_add,
                                             imitation_learning=True)
            env = Monitor(env, logger.get_dir() and
                          os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)
            return env
        return thunk
    set_global_seeds(seed)

    # If not wanting vectorized envs
    if no_vec and settings["totalCpus"] == 1:
        return make_sb_env(0)()

    # When using one environment, no need to start subprocesses
    if settings["totalCpus"] == 1 or not use_subprocess:
        return DummyVecEnv([make_sb_env(i + start_index) for i in range(settings["totalCpus"])])

    return SubprocVecEnv([make_sb_env(i + start_index) for i in range(settings["totalCpus"])],
                         start_method=start_method)
