import sys
import os
import time
import yaml
import argparse
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../'))

from make_stable_baselines_env import make_stable_baselines_env
from wrappers.tektag_rew_wrap import TektagRoundEndChar2Penalty, TektagHealthBarUnbalancePenalty

from stable_baselines import PPO2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFile', type=str, required=True, help='Training configuration file')
    opt = parser.parse_args()
    print(opt)

    # Read the cfg file
    yaml_file = open(opt.cfgFile)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Params = ", params)
    yaml_file.close()

    time_dep_seed = int((time.time() - int(time.time() - 0.5)) * 1000)

    model_folder = os.path.join(base_path, params["settings_game_id"] + params["model_folder"])

    # Settings
    settings = {}
    settings["gameId"] = params["settings_game_id"]
    settings["step_ratio"] = params["settings_step_ratio"]
    settings["frame_shape"] = params["settings_frame_shape"]
    settings["player"] = "P1"  # P1 / P2

    settings["characters"] = params["settings_characters"]

    settings["difficulty"] = params["settings_difficulty"]
    settings["charOutfits"] = [2, 2]

    settings["continueGame"] = 0.0
    settings["showFinal"] = False

    settings["action_space"] = params["settings_action_space"]
    settings["attack_but_combination"] = params["settings_attack_but_combination"]

    # Wrappers settings
    wrappers_settings = {}
    wrappers_settings["noOpMax"] = 0
    wrappers_settings["rewardNormalization"] = True
    wrappers_settings["clipRewards"] = False
    wrappers_settings["frame_stack"] = params["wrappers_settings_frame_stack"]
    wrappers_settings["dilation"] = params["wrappers_settings_dilation"]
    wrappers_settings["actions_stack"] = params["wrappers_settings_actions_stack"]
    wrappers_settings["scale"] = True
    wrappers_settings["scaleMod"] = 0

    # Additional obs key list
    key_to_add = []
    for key in params["key_to_add"]:
        key_to_add.append(key)

    env, numEnv = make_stable_baselines_env(time_dep_seed, settings, wrappers_settings,
                                            key_to_add=key_to_add, no_vec=True)

    # Load the trained agent
    model_checkpoint = params["ppo_model_checkpoint"]
    model = PPO2.load(os.path.join(model_folder, model_checkpoint))

    obs = env.reset()
    cumulative_rew = 0.0

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)
        cumulative_rew += reward

        if done:
            print("Cumulative Rew =", cumulative_rew)
            cumulative_rew = 0.0
            obs = env.reset()

    # Close the environment
    env.close()
