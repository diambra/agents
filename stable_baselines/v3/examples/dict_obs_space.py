import os
import sys
import diambra.arena

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.utils import set_random_seed

# Make Stable Baselines Env function
def make_stable_baselines_env(game_id, env_settings, wrappers_settings=None,
                              use_subprocess=True, seed=0):
    """
    Create a wrapped VecEnv.
    :param game_id: (str) the game environment ID
    :param env_settings: (dict) parameters for DIAMBRA Arena environment
    :param wrappers_settings: (dict) parameters for environment
                              wraping function
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
            return env
        return _init
    set_random_seed(seed)

    # When using one environment, no need to start subprocesses
    if num_envs == 1 or not use_subprocess:
        env = DummyVecEnv([make_sb_env(i) for i in range(num_envs)])
    else:
        env = SubprocVecEnv([make_sb_env(i) for i in range(num_envs)])

    return env, num_envs

if __name__ == '__main__':

    # Settings
    settings = {}
    settings["frame_shape"] = [128, 128, 1]
    settings["characters"] = [["Kasumi"], ["Kasumi"]]

    # Wrappers Settings
    wrappers_settings = {}
    wrappers_settings["reward_normalization"] = True
    wrappers_settings["actions_stack"] = 12
    wrappers_settings["frame_stack"] = 5
    wrappers_settings["flatten"] = True
    wrappers_settings["filter_keys"] = ["stage", "P1_ownHealth", "P1_oppHealth", "P1_ownSide",
                                        "P1_oppSide", "P1_oppChar", "P1_actions_move", "P1_actions_attack"]

    # Create environment
    env, num_envs = make_stable_baselines_env("doapp", settings, wrappers_settings)
    print("Activated {} environment(s)".format(num_envs))

    print("Observation space =", env.observation_space)
    print("Act_space =", env.action_space)

    # Instantiate the agent
    model = PPO('MultiInputPolicy', env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=1000)

    # Enjoy trained agent
    observation = env.reset()
    cumulative_reward = [0.0 for _ in range(num_envs)]
    while True:
        env.render()

        action, _state = model.predict(observation, deterministic=True)

        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if any(x != 0 for x in reward):
            print("Cumulative reward(s) =", cumulative_reward)

        if done.any():
            observation = env.reset()
            break

    env.close()
