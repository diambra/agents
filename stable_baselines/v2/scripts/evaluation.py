import sys
import os
import time
import yaml
import json
import argparse
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../'))
from make_stable_baselines_env import make_stable_baselines_env

from stable_baselines import PPO2

if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfgFile', type=str, required=True, help='Configuration file')
        parser.add_argument('--trainedModel', type=str, required=True, help='Model checkpoint')
        opt = parser.parse_args()
        print(opt)

        # Read the cfg file
        yaml_file = open(opt.cfgFile)
        params = yaml.load(yaml_file, Loader=yaml.FullLoader)
        print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
        yaml_file.close()

        time_dep_seed = int((time.time() - int(time.time() - 0.5)) * 1000)

        model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                    params["folders"]["model_name"], "model")

        # Settings
        settings = params["settings"]
        settings["Player"] = "P1"

        # Wrappers Settings
        wrappers_settings = params["wrappers_settings"]
        wrappers_settings["reward_normalization"] = False

        # Additional obs key list
        key_to_add = params["key_to_add"]

        env, num_env = make_stable_baselines_env(time_dep_seed, settings, wrappers_settings,
                                                 key_to_add=key_to_add, no_vec=True)

        # Load the trained agent
        model_path = os.path.join(model_folder, opt.trainedModel)
        model = PPO2.load(model_path)

        obs = env.reset()

        while True:

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            if done:
                obs = env.reset()

        # Close the environment
        env.close()

        print("COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(e)
        print("ERROR, ABORTED.")
