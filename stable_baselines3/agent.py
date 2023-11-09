import os
import yaml
import json
import argparse
from diambra.arena import Roles, SpaceTypes, load_settings_flat_dict
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download, login as huggingface_hub_login

"""This is an example agent based on stable baselines 3.

Usage:
diambra run python stable_baselines3/agent.py --cfgFile $PWD/stable_baselines3/cfg_files/doapp/sr6_128x4_das_nc.yaml --trainedModel "model_name"
"""

def main(repo, cfg_file, trained_model, test=False):

    if os.getenv("HF_TOKEN"):
        huggingface_hub_login(os.getenv("HF_TOKEN").strip())

    config_path = hf_hub_download(repo_id=repo, filename=cfg_file)
    # Read the cfg file
    yaml_file = open(config_path)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    model_repo_path = os.path.join(params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model", trained_model)
    
    model_path = hf_hub_download(repo_id=repo, filename=model_repo_path)


    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])
    settings.role = Roles.P1

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])
    wrappers_settings.normalize_reward = False

    # Create environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, no_vec=True)
    print("Activated {} environment(s)".format(num_envs))

    # Load the trained agent
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
    parser.add_argument("--repo", type=str, required=True, help="Repository name")
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument("--trainedModel", type=str, default="model.zip", help="Model checkpoint")
    parser.add_argument("--test", type=int, default=0, help="Test mode")
    opt = parser.parse_args()
    print(opt)

    main(opt.repo, opt.cfgFile, opt.trainedModel, bool(opt.test))
