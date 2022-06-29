import sys
import os
import time
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../../'))
from make_stable_baselines_env import make_stable_baselines_env
from sbUtils import linear_schedule, AutoSave, ModelCfgSave
from custom_policies.custom_cnn_policy import CustCnnPolicy, local_nature_cnn_small
from stable_baselines import PPO2

if __name__ == '__main__':
    time_dep_seed = int((time.time()-int(time.time()-0.5))*1000)

    model_folder = os.path.join(base_path, "model/")
    tensor_board_folder = os.path.join(base_path, "TB/")

    os.makedirs(model_folder, exist_ok=True)

    # Settings
    settings = {}
    settings["game_id"] = "doapp"
    settings["step_ratio"] = 6
    settings["frame_shape"] = [128, 128, 1]
    settings["player"] = "Random"  # P1 / P2

    settings["characters"] = [["Kasumi"], ["Kasumi"]]

    settings["difficulty"] = 4
    settings["char_outfits"] = [2, 2]

    settings["continue_game"] = 0.0
    settings["show_final"] = False

    settings["action_space"] = "discrete"
    settings["attack_but_combination"] = False

    # Wrappers Settings
    wrappers_settings = {}
    wrappers_settings["no_op_max"] = 0
    wrappers_settings["reward_normalization"] = True
    wrappers_settings["clip_rewards"] = False
    wrappers_settings["frame_stack"] = 4
    wrappers_settings["dilation"] = 1
    wrappers_settings["actions_stack"] = 12
    wrappers_settings["scale"] = True
    wrappers_settings["scale_mod"] = 0

    # Additional obs key list
    key_to_add = []
    key_to_add.append("actions")

    key_to_add.append("ownHealth")
    key_to_add.append("oppHealth")

    key_to_add.append("ownSide")
    key_to_add.append("oppSide")
    key_to_add.append("stage")

    env, num_env = make_stable_baselines_env(time_dep_seed, settings, wrappers_settings,
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
    policy_kwargs["n_add_info"] = n_actions_stack*(n_actions[0]+n_actions[1]) +\
        len(key_to_add)-1
    policy_kwargs["layers"] = [64, 64]

    policy_kwargs["cnn_extractor"] = local_nature_cnn_small

    print("n_actions =", n_actions)
    print("n_char =", n_char)
    print("n_add_info =", policy_kwargs["n_add_info"])

    # PPO param
    gamma = 0.94
    model_checkpoint = "346M"

    '''
    learning_rate = linear_schedule(2.5e-4, 2.5e-6)
    cliprange = linear_schedule(0.15, 0.025)
    cliprange_vf = cliprange
    # Initialize the model
    model = PPO2(CustCnnPolicy, env, verbose=1,
                 gamma=gamma, nminibatches=8, noptepochs=4, n_steps=128,
                 learning_rate=learning_rate, cliprange=cliprange,
                 cliprange_vf=cliprange_vf, policy_kwargs=policy_kwargs,
                 tensorboard_log=tensor_board_folder)
    #OR
    '''
    learning_rate = linear_schedule(5.0e-5, 2.5e-6)
    cliprange = linear_schedule(0.075, 0.025)
    cliprange_vf = cliprange
    # Load the trained agent
    model = PPO2.load(os.path.join(model_folder, model_checkpoint), env=env,
                      policy_kwargs=policy_kwargs, gamma=gamma,
                      learning_rate=learning_rate,
                      cliprange=cliprange, cliprange_vf=cliprange_vf,
                      tensorboard_log=tensor_board_folder)

    print("Model discount factor = ", model.gamma)

    # Create the callback: autosave every USER DEF steps
    auto_save_callback = AutoSave(check_freq=1000000, num_env=num_env,
                                  save_path=os.path.join(model_folder, model_checkpoint + "_"))

    # Train the agent
    time_steps = 30000000
    model.learn(total_timesteps=time_steps, callback=auto_save_callback)

    # Save the agent
    model_path = os.path.join(model_folder, "386M")
    model.save(model_path)
    # Save the correspondent CFG file
    ModelCfgSave(model_path, "PPOSmall", n_actions, char_names,
                 settings, wrappers_settings, key_to_add)

    # Close the environment
    env.close()
