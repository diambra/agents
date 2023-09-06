import os
import yaml
import json
import argparse
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO

"""This is an example agent based on stable baselines 3.

Usage:
diambra run python stable_baselines3/agent.py --cfgFile $PWD/stable_baselines3/cfg_files/doapp/sr6_128x4_das_nc.yaml --trainedModel "model_name"
"""

def main(cfg_file, trained_model, test=False):
    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")

    # Settings
    settings = params["settings"]
    settings["role"] = "P1"

    # Wrappers Settings
    wrappers_settings = params["wrappers_settings"]
    wrappers_settings["reward_normalization"] = False

    # Create environment
    env, num_envs = make_sb3_env(settings["game_id"], settings, wrappers_settings, no_vec=True)
    print("Activated {} environment(s)".format(num_envs))

    print("Observation space =", env.observation_space)
    print("Act_space =", env.action_space)

    # Load the trained agent
    model_path = os.path.join(model_folder, trained_model)
    agent = PPO.load(model_path)

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    obs, info = env.reset()

    while True:
        action, _ = agent.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action.tolist())

        if terminated or truncated:
            obs, info = env.reset()
            if info["env_done"] or test is True:
                break

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument("--trainedModel", type=str, default="model", help="Model checkpoint")
    parser.add_argument("--test", type=int, default=0, help="Test mode")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile, opt.trainedModel, bool(opt.test))
