import os
import yaml
import json
import argparse
from diambra.arena import Roles, SpaceTypes, load_settings_flat_dict
from custom_wrappers import RamStatesToChannel, SplitActionsInMoveAndAttack
from diambra.arena.stable_baselines.make_sb_env import make_sb_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines.sb_utils import show_obs
from stable_baselines import PPO2

"""This is an example agent based on stable baselines.
Usage:
diambra run python stable_baselines/agent.py --cfgFile $PWD/stable_baselines/cfg_files/doapp/sr6_128x4_das_nc.yaml --trainedModel "model_name"
"""

def main(cfg_file, trained_model):
    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")

    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])
    settings.role = Roles.P1

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])
    wrappers_settings.normalize_reward = False

    # Additional obs key list
    wrappers_settings.wrappers = [[SplitActionsInMoveAndAttack, {}],
                                  [RamStatesToChannel, {"ram_states": params["ram_states"]}]]

    env, num_env = make_sb_env(settings.game_id, settings, wrappers_settings, no_vec=True)

    # Load the trained agent
    model_path = os.path.join(model_folder, trained_model)
    agent = PPO2.load(model_path)

    obs = env.reset()
    #show_obs(obs, params["ram_states"], env.n_actions, wrappers_settings["actions_stack"], env.env_info.characters_info.char_list, True, 0)

    while True:
        action, _ = agent.predict(obs, deterministic=False)

        obs, reward, done, info = env.step(action)
        #show_obs(obs, params["ram_states"], env.n_actions, wrappers_settings["actions_stack"], env.env_info.characters_info.char_list, True, 0)

        if done:
            obs = env.reset()
            if info["env_done"]:
                break

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument("--trainedModel", type=str, default="model", help="Model checkpoint")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile, opt.trainedModel)
