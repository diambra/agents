import sys
import os
import time

if __name__ == '__main__':
    time_dep_seed = int((time.time()-int(time.time()-0.5))*1000)

    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_path, '../../'))

    model_folder = os.path.join(base_path, "model_30Hz/")

    from make_stable_baselines_env import make_stable_baselines_env
    from sb_utils import show_obs

    from stable_baselines import PPO2

    # Settings
    settings = {}
    settings["game_id"] = "sfiii3n"
    settings["step_ratio"] = 2
    settings["frame_shape"] = [128, 128, 1]
    settings["player"] = "P1"  # P1 / P2

    settings["characters"] = [["Ryu"], ["Ryu"]]

    settings["difficulty"] = 6
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
    wrappers_settings["dilation"] = 3
    wrappers_settings["actions_stack"] = 36
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

    # Load the trained agent
    model = PPO2.load(os.path.join(model_folder, "600M"))

    obs = env.reset()
    cumulative_rew = 0.0

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)
        #show_obs(obs, keyToAdd, env.key_to_add_count, wrapper_kwargs["actions_stack"], env.n_actions,
        #         wait_key, True, env.char_names, False, [0])
        cumulative_rew += reward

        if done:
            stage = 1
            print("Cumulative Rew =", cumulative_rew)
            cumulative_rew = 0.0
            obs = env.reset()

    # Close the environment
    env.close()
