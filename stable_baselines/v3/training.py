import os
import sys
import time
import yaml
import json
import argparse
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
from make_stable_baselines_env import make_stable_baselines_env

from stable_baselines3 import PPO

if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfgFile', type=str, required=True, help='Configuration file')
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
        tensor_board_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                           params["folders"]["model_name"], "tb")

        os.makedirs(model_folder, exist_ok=True)

        # Settings
        settings = params["settings"]

        # Wrappers Settings
        wrappers_settings = params["wrappers_settings"]

        # Create environment
        env, num_env = make_stable_baselines_env(params["settings"]["game_id"], settings, wrappers_settings, seed=time_dep_seed)
        print("Activated {} environment(s)".format(num_envs))

        print("Observation space =", env.observation_space)
        print("Act_space =", env.action_space)

        # Instantiate the agent
        model = PPO('MultiInputPolicy', env, verbose=1)

        # Train the agent
        time_steps = ppo_settings["time_steps"]
        model.learn(total_timesteps=time_steps)

        # Save the agent
        new_model_checkpoint = str(int(model_checkpoint[:-1]) + time_steps) + "M"
        model_path = os.path.join(model_folder, new_model_checkpoint)
        model.save(model_path)

        # Close the environment
        env.close()

        print("COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(e)
        print("ERROR, ABORTED.")
        sys.exit(1)
