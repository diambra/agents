import sys
import os
import time
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../.'))
from make_stable_baselines_env import make_stable_baselines_env
from stable_baselines import PPO2

if __name__ == '__main__':

    time_dep_seed = int((time.time() - int(time.time() - 0.5)) * 1000)

    model_folder = os.path.join(base_path, "models/")

    # Settings
    settings = {
        "game_id": "doapp",
        "env_address": "0.0.0.0:50052",
        "player": "Random",
        "step_ratio": 6,
        "frame_shape": [128, 128, 1],
        "hardcore": False,
        "difficulty": 4,
        "characters": [["Kasumi"], ["Kasumi"]],
        "char_outfits": [2, 2],
        "action_space": "discrete",
        "attack_but_combination": False
    }

    # Wrappers Settings
    wrappers_settings = {
        "reward_normalization": True,
        "frame_stack": 4,
        "actions_stack": 12,
        "scale": True
    }

    # Additional obs key list
    key_to_add = []
    key_to_add.append("actions")

    key_to_add.append("ownHealth")
    key_to_add.append("oppHealth")

    key_to_add.append("ownSide")
    key_to_add.append("oppSide")
    key_to_add.append("stage")

    env, num_env = make_stable_baselines_env(time_dep_seed, settings, wrappers_settings,
                                            key_to_add=key_to_add, no_vec=True)

    # Load the trained agent
    model = PPO2.load(os.path.join(model_folder, "doappSmall25M"))

    obs = env.reset()

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

    # Close the environment
    env.close()
