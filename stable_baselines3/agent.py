import os
import time
import yaml
import json
import argparse
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO

"""This is a example agent based on stable baselines 3.

Usage:
diambra run -g python stable_baselines/agent.py --cfgFile $PWD/stable_baselines/cfg_files/doapp/sr6_128x4_das_nc.yaml --trainedModel "512M"
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument("--trainedModel", type=str, required=True, help="Model checkpoint")
    opt = parser.parse_args()
    print(opt)

    # Read the cfg file
    yaml_file = open(opt.cfgFile)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    time_dep_seed = int((time.time() - int(time.time() - 0.5)) * 1000)

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")

    # Settings
    settings = params["settings"]
    settings["player"] = "P1"

    # Wrappers Settings
    wrappers_settings = params["wrappers_settings"]
    wrappers_settings["reward_normalization"] = False

    # Create environment
    env, num_envs = make_sb3_env(params["settings"]["game_id"], settings, wrappers_settings, seed=time_dep_seed, no_vec=True)
    print("Activated {} environment(s)".format(num_envs))

    print("Observation space =", env.observation_space)
    print("Act_space =", env.action_space)

    # Load the trained agent
    model_path = os.path.join(model_folder, opt.trainedModel)
    agent = PPO.load(model_path, env=env)

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    obs = env.reset()

    while True:

        action, _ = agent.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()
            if info["env_done"]:
                break

    # Close the environment
    env.close()
