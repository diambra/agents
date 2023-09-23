import os
import yaml
import json
import argparse
from custom_wrappers import RamStatesToChannel, SplitActionsInMoveAndAttack
from diambra.arena import SpaceTypes, load_settings_flat_dict
from diambra.arena.stable_baselines.make_sb_env import make_sb_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines.sb_utils import linear_schedule, AutoSave
from custom_policies.custom_cnn_policy import CustCnnPolicy, local_nature_cnn_small
from stable_baselines import PPO2

def main(cfg_file):
    # Read the cfg file
    yaml_file = open(cfg_file)
    params = yaml.load(yaml_file, Loader=yaml.FullLoader)
    print("Config parameters = ", json.dumps(params, sort_keys=True, indent=4))
    yaml_file.close()

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                params["folders"]["model_name"], "model")
    tensor_board_folder = os.path.join(base_path, params["folders"]["parent_dir"], params["settings"]["game_id"],
                                       params["folders"]["model_name"], "tb")

    os.makedirs(model_folder, exist_ok=True)

    # Settings
    params["settings"]["action_space"] = SpaceTypes.DISCRETE if params["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, params["settings"])

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, params["wrappers_settings"])

    # Additional obs key list
    wrappers_settings.wrappers = [[SplitActionsInMoveAndAttack, {}],
                                  [RamStatesToChannel, {"ram_states": params["ram_states"]}]]

    env, num_envs = make_sb_env(settings.game_id, settings, wrappers_settings, use_subprocess=True)

    # Policy param
    policy_kwargs = params["policy_kwargs"]

    if params["use_small_cnn"] is True:
        policy_kwargs["cnn_extractor"] = local_nature_cnn_small

    # PPO settings
    ppo_settings = params["ppo_settings"]
    gamma = ppo_settings["gamma"]
    model_checkpoint = ppo_settings["model_checkpoint"]

    learning_rate = linear_schedule(ppo_settings["learning_rate"][0], ppo_settings["learning_rate"][1])
    cliprange = linear_schedule(ppo_settings["cliprange"][0], ppo_settings["cliprange"][1])
    cliprange_vf = cliprange
    nminibatches = ppo_settings["nminibatches"]
    noptepochs = ppo_settings["noptepochs"]
    n_steps = ppo_settings["n_steps"]

    if model_checkpoint == "0":
        # Initialize the agent
        agent = PPO2(CustCnnPolicy, env, verbose=1,
                        gamma=gamma, nminibatches=nminibatches,
                        noptepochs=noptepochs, n_steps=n_steps,
                        learning_rate=learning_rate, cliprange=cliprange,
                        cliprange_vf=cliprange_vf, policy_kwargs=policy_kwargs,
                        tensorboard_log=tensor_board_folder)
    else:

        # Load the trained agent
        agent = PPO2.load(os.path.join(model_folder, model_checkpoint), env=env,
                            policy_kwargs=policy_kwargs, gamma=gamma,
                            learning_rate=learning_rate,
                            cliprange=cliprange, cliprange_vf=cliprange_vf,
                            tensorboard_log=tensor_board_folder)

    print("Model discount factor = ", agent.gamma)

    # Create the callback: autosave every USER DEF steps
    autosave_freq = ppo_settings["autosave_freq"]
    auto_save_callback = AutoSave(check_freq=autosave_freq, num_envs=num_envs,
                                  save_path=model_folder, filename_prefix=model_checkpoint + "_")

    # Train the agent
    time_steps = ppo_settings["time_steps"]
    agent.learn(total_timesteps=time_steps, callback=auto_save_callback)

    # Save the agent
    new_model_checkpoint = str(int(model_checkpoint) + time_steps)
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.save(model_path)

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFile', type=str, required=True, help='Configuration file')
    opt = parser.parse_args()
    print(opt)

    main(opt.cfgFile)
