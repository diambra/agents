import os
import sys
import time
import yaml
import json
import argparse
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
from make_stable_baselines_env import make_stable_baselines_env
from sb3_utils import linear_schedule, AutoSave

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
        env, num_envs = make_stable_baselines_env(params["settings"]["game_id"], settings, wrappers_settings, seed=time_dep_seed)
        print("Activated {} environment(s)".format(num_envs))

        print("Observation space =", env.observation_space)
        print("Act_space =", env.action_space)

        # Policy param
        policy_kwargs = None  # params["policy_kwargs"] temporarily deactivated

        # PPO settings
        ppo_settings = params["ppo_settings"]
        gamma = ppo_settings["gamma"]
        model_checkpoint = ppo_settings["model_checkpoint"]

        learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
        clip_range = linear_schedule(ppo_settings["cliprange"][0], ppo_settings["cliprange"][1])
        clip_range_vf = clip_range
        batch_size = ppo_settings["batch_size"]
        n_epochs = ppo_settings["n_epochs"]
        n_steps = ppo_settings["n_steps"]

        if model_checkpoint == "0M":
            # Initialize the model
            model = PPO('MultiInputPolicy', env, verbose=1,
                        gamma=gamma, batch_size=batch_size,
                        n_epochs=n_epochs, n_steps=n_steps,
                        learning_rate=learning_rate, clip_range=clip_range,
                        clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                        tensorboard_log=tensor_board_folder)
        else:
            # Load the trained agent
            model = PPO.load(os.path.join(model_folder, model_checkpoint), env=env,
                             gamma=gamma, learning_rate=learning_rate, clip_range=clip_range,
                             clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                             tensorboard_log=tensor_board_folder)

        print("Model discount factor = ", model.gamma)

        # Create the callback: autosave every USER DEF steps
        autosave_freq = ppo_settings["autosave_freq"]
        auto_save_callback = AutoSave(check_freq=autosave_freq, num_envs=num_envs,
                                      save_path=os.path.join(model_folder, model_checkpoint + "_"))

        # Train the agent
        time_steps = ppo_settings["time_steps"]
        model.learn(total_timesteps=time_steps, callback=auto_save_callback)

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
