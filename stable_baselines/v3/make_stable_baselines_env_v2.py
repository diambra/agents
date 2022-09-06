import os
import sys
import diambra.arena

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines import logger
from stable_baselines.bench import Monitor

# Make Stable Baselines Env function
def make_stable_baselines_env(game_id, env_settings, wrappers_settings=None,
                              start_index=0, allow_early_resets=True,
                              start_method=None, no_vec=False,
                              use_subprocess=True, seed=0):
    """
    Create a wrapped, monitored VecEnv.
    :param game_id: (str) the game environment ID
    :param env_settings: (dict) parameters for DIAMBRA Arena environment
    :param wrappers_settings: (dict) parameters for environment
                              wraping function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
                        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or
                          `DummyVecEnv` when
    :param no_vec: (bool) Whether to avoid usage of Vectorized Env or not.
                   Default: False
    :param seed: (int) initial seed for RNG
    :return: (VecEnv) The diambra environment
    """

    env_addresses = os.getenv("DIAMBRA_ENVS", "").split()
    if len(env_addresses) == 0:
        raise Exception("ERROR: Running script without DIAMBRA CLI.")
        sys.exit(1)

    num_envs = len(env_addresses)

    def make_sb_env(rank):
        def _init():
            env = diambra.arena.make(game_id, env_settings, wrappers_settings,
                                     seed=seed + rank, rank=rank)

            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)
            return env
        return _init
    set_random_seed(seed)

    # If not wanting vectorized envs
    if no_vec and num_envs == 1:
        env = make_sb_env(0)()
    else:
        # When using one environment, no need to start subprocesses
        if num_envs == 1 or not use_subprocess:
            env = DummyVecEnv([make_sb_env(i + start_index) for i in range(num_envs)])
        else:
            env = SubprocVecEnv([make_sb_env(i + start_index) for i in range(num_envs)],
                             start_method=start_method)

    return env, num_envs
