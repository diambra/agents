import sys
import os
import time
import yaml
import argparse
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../'))
from make_stable_baselines_env import make_stable_baselines_env
from wrappers.tektag_rew_wrap import TektagRoundEndChar2Penalty, TektagHealthBarUnbalancePenalty

from sb_utils import linear_schedule, AutoSave, model_cfg_save
from custom_policies.custom_cnn_policy import CustCnnPolicy, local_nature_cnn_small

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
    tensor_board_folder = os.path.join(base_path, params["settings_game_id"] + params["tensor_board_folder"])

    os.makedirs(model_folder, exist_ok=True)

    # Settings
    settings = {}
    settings["game_id"] = params["settings_game_id"]
    settings["step_ratio"] = params["settings_step_ratio"]
    settings["frame_shape"] = params["settings_frame_shape"]
    settings["player"] = "Random"  # P1 / P2

    settings["characters"] = params["settings_characters"]

    settings["difficulty"] = params["settings_difficulty"]
    settings["char_outfits"] = [2, 2]

    settings["continue_game"] = params["settings_continue_game"]
    settings["show_final"] = False

    settings["action_space"] = params["settings_action_space"]
    settings["attack_but_combination"] = params["settings_attack_but_combination"]

    # Wrappers Settings
    wrappers_settings = {}
    wrappers_settings["no_op_max"] = 0
    wrappers_settings["reward_normalization"] = True
    wrappers_settings["clip_rewards"] = False
    wrappers_settings["frame_stack"] = params["wrappers_settings_frame_stack"]
    wrappers_settings["dilation"] = params["wrappers_settings_dilation"]
    wrappers_settings["actions_stack"] = params["wrappers_settings_actions_stack"]
    wrappers_settings["scale"] = True
    wrappers_settings["scale_mod"] = 0

    # Additional custom wrappers
    custom_wrappers = None
    if params["custom_wrappers"] is True:
        custom_wrappers = [TektagRoundEndChar2Penalty, TektagHealthBarUnbalancePenalty]

    # Additional obs key list
    key_to_add = []
    for key in params["key_to_add"]:
        key_to_add.append(key)

    env, num_env = make_stable_baselines_env(time_dep_seed, settings, wrappers_settings,
                                             custom_wrappers=custom_wrappers,
                                             key_to_add=key_to_add, use_subprocess=True)

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

    # Policy param
    n_actions = env.get_attr("n_actions")[0][0]
    n_actions_stack = env.get_attr("n_actions_stack")[0]
    n_char = env.get_attr("number_of_characters")[0]
    char_names = env.get_attr("char_names")[0]

    policy_kwargs = {}
    policy_kwargs["n_add_info"] = params["policy_kwargs_n_add_info"]
    policy_kwargs["layers"] = params["policy_kwargs_layers"]

    if params["policy_kwargs_use_small_cnn"] is True:
        policy_kwargs["cnn_extractor"] = local_nature_cnn_small

    print("n_actions =", n_actions)
    print("n_char =", n_char)
    print("n_add_info =", policy_kwargs["n_add_info"])

    # PPO param
    gamma = params["ppo_gamma"]
    model_checkpoint = params["ppo_model_checkpoint"]

    learning_rate = linear_schedule(params["ppo_learning_rate"][0], params["ppo_learning_rate"][1])
    cliprange = linear_schedule(params["ppo_cliprange"][0], params["ppo_cliprange"][1])
    cliprange_vf = cliprange
    nminibatches = params["ppo_nminibatches"]
    noptepochs = params["ppo_noptepochs"]
    n_steps = params["ppo_n_steps"]

    if model_checkpoint == "0M":
        # Initialize the model
        model = PPO2(CustCnnPolicy, env, verbose=1,
                     gamma=gamma, nminibatches=nminibatches,
                     noptepochs=noptepochs, n_steps=n_steps,
                     learning_rate=learning_rate, cliprange=cliprange,
                     cliprange_vf=cliprange_vf, policy_kwargs=policy_kwargs,
                     tensorboard_log=tensor_board_folder)
    else:

        # Load the trained agent
        model = PPO2.load(os.path.join(model_folder, model_checkpoint), env=env,
                          policy_kwargs=policy_kwargs, gamma=gamma,
                          learning_rate=learning_rate,
                          cliprange=cliprange, cliprange_vf=cliprange_vf,
                          tensorboard_log=tensor_board_folder)

    print("Model discount factor = ", model.gamma)

    # Create the callback: autosave every USER DEF steps
    autosave_freq = params["ppo_autosave_freq"]
    auto_save_callback = AutoSave(check_freq=autosave_freq, num_env=num_env,
                                  save_path=os.path.join(model_folder, model_checkpoint + "_"))

    # Train the agent
    time_steps = params["ppo_time_steps"]
    model.learn(total_timesteps=time_steps, callback=auto_save_callback)

    # Save the agent
    new_model_checkpoint = str(int(model_checkpoint[:-1]) + time_steps) + "M"
    model_path = os.path.join(model_folder, new_model_checkpoint)
    model.save(model_path)

    # Close the environment
    env.close()
