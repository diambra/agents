import os
import sys
import time
import yaml
import json
import argparse
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from stable_baselines3 import PPO

# diambra run python stable_baselines3/training.py --cfgFile $PWD/stable_baselines3/cfg_files/doapp/sr6_128x4_das_nc.yaml

if __name__ == "__main__":

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
    base_path = os.path.dirname(os.path.abspath(__file__))
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
    env, num_envs = make_sb3_env(params["settings"]["game_id"], settings, wrappers_settings, seed=time_dep_seed)
    print("Activated {} environment(s)".format(num_envs))

    print("Observation space =", env.observation_space)
    print("Act_space =", env.action_space)

    # Policy param
    policy_kwargs = params["policy_kwargs"]

    # PPO settings
    ppo_settings = params["ppo_settings"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]

    learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    clip_range = linear_schedule(ppo_settings["clip_range"][0], ppo_settings["clip_range"][1])
    clip_range_vf = clip_range
    batch_size = ppo_settings["batch_size"]
    n_epochs = ppo_settings["n_epochs"]
    n_steps = ppo_settings["n_steps"]

    if model_checkpoint == "0M":
        # Initialize the agent
        agent = PPO('MultiInputPolicy', env, verbose=1,
                    gamma=gamma, batch_size=batch_size,
                    n_epochs=n_epochs, n_steps=n_steps,
                    learning_rate=learning_rate, clip_range=clip_range,
                    clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                    tensorboard_log=tensor_board_folder)
    else:
        # Load the trained agent
        agent = PPO.load(os.path.join(model_folder, model_checkpoint), env=env,
                            gamma=gamma, learning_rate=learning_rate, clip_range=clip_range,
                            clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                            tensorboard_log=tensor_board_folder)


    # Print policy network architecture
    print("Policy architecure:")
    print(agent.policy)

    # Create the callback: autosave every USER DEF steps
    autosave_freq = ppo_settings["autosave_freq"]
    auto_save_callback = AutoSave(check_freq=autosave_freq, num_envs=num_envs,
                                    save_path=os.path.join(model_folder, model_checkpoint + "_"))

    # Train the agent
    time_steps = ppo_settings["time_steps"]
    agent.learn(total_timesteps=time_steps, callback=auto_save_callback)

    # Save the agent
    new_model_checkpoint = str(int(model_checkpoint[:-1]) + time_steps) + "M"
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.save(model_path)

    # Close the environment
    env.close()
