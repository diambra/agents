import sys
import os
import time
import yaml
import json
import argparse
from diambra.arena.stable_baselines.make_sb_env import make_sb_env
from diambra.arena.stable_baselines.wrappers.tektag_rew_wrap import TektagRoundEndChar2Penalty, TektagHealthBarUnbalancePenalty
from diambra.arena.stable_baselines.sb_utils import linear_schedule, AutoSave
from diambra.arena.stable_baselines.custom_policies.custom_cnn_policy import CustCnnPolicy, local_nature_cnn_small
from stable_baselines import PPO2

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

        # Additional custom wrappers
        custom_wrappers = None
        if params["custom_wrappers"] is True:
            custom_wrappers = [TektagRoundEndChar2Penalty, TektagHealthBarUnbalancePenalty]

        # Additional obs key list
        key_to_add = params["key_to_add"]

        env, num_env = make_sb_env(time_dep_seed, settings, wrappers_settings,
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

        if model_checkpoint == "0M":
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
        auto_save_callback = AutoSave(check_freq=autosave_freq, num_env=num_env,
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

        print("COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(e)
        print("ERROR, ABORTED.")
        sys.exit(1)
