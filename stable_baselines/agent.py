import os
import time
import yaml
import json
import argparse
from diambra.arena.stable_baselines.make_sb_env import make_sb_env
from stable_baselines import PPO2

"""This is an example agent based on stable baselines.

Usage:
diambra run python stable_baselines/agent.py --cfgFile $PWD/stable_baselines/cfg_files/doapp/sr6_128x4_das_nc.yaml --trainedModel "model_name"
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument("--trainedModel", type=str, default="model", help="Model checkpoint")
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

    # Additional obs key list
    key_to_add = params["key_to_add"]

    env, num_env = make_sb_env(time_dep_seed, settings, wrappers_settings, key_to_add=key_to_add, no_vec=True)

    print("Obs_space = ", env.observation_space)
    print("Obs_space type = ", env.observation_space.dtype)
    print("Obs_space high = ", env.observation_space.high)
    print("Obs_space low = ", env.observation_space.low)

    print("Act_space = ", env.action_space)
    print("Act_space type = ", env.action_space.dtype)
    if settings["action_space"] == "multi_discrete":
        print("Act_space n = ", env.action_space.nvec)
    else:
        print("Act_space n = ", env.action_space.n)

    # Load the trained agent
    model_path = os.path.join(model_folder, opt.trainedModel)
    agent = PPO2.load(model_path)

    obs = env.reset()

    while True:

        action, _ = agent.predict(obs, deterministic=False)

        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()
            if info["env_done"]:
                break

    # Close the environment
    env.close()
