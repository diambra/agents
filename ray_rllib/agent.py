import os
import time
import yaml
import json
import argparse
import diambra.arena
from diambra.arena.ray_rllib.make_ray_env import DiambraArena, preprocess_ray_config
from ray.rllib.algorithms.ppo import PPO

# Reference: https://github.com/ray-project/ray/blob/ray-2.0.0/rllib/examples/inference_and_serving/policy_inference_after_training.py

"""This is an example agent based on RL Lib.

Usage:
diambra run python agent.py --trainedModel /absolute/path/to/checkpoint/ --envSpaces /absolute/path/to/environment/spaces/descriptor/
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainedModel", type=str, required=True, help="Model path")
    parser.add_argument("--envSpaces", type=str, required=True, help="Environment spaces descriptor file path")
    opt = parser.parse_args()
    print(opt)

    time_dep_seed = int((time.time() - int(time.time() - 0.5)) * 1000)

    # Settings
    settings = {}
    settings["frame_shape"] = (84, 84, 1)
    settings["characters"] = ("Kasumi")

    # Wrappers Settings
    wrappers_settings = {}
    wrappers_settings["reward_normalization"] = True
    wrappers_settings["actions_stack"] = 12
    wrappers_settings["frame_stack"] = 5
    wrappers_settings["scale"] = True
    wrappers_settings["process_discrete_binary"] = True

    config = {
        # Define and configure the environment
        "env": DiambraArena,
        "env_config": {
            "game_id": "doapp",
            "settings": settings,
            "wrappers_settings": wrappers_settings,
            "load_spaces_from_file": True,
            "env_spaces_file_name": opt.envSpaces,
        },
        "num_workers": 0,
        "train_batch_size": 200,
        "framework": "torch",
    }

    # Update config file
    config = preprocess_ray_config(config)

    # Load the trained agent
    agent = PPO(config=config)
    agent.restore(opt.trainedModel)
    print("Agent loaded")

    # Print the agent policy architecture
    print("Policy architecture =\n{}".format(agent.get_policy().model))

    env = diambra.arena.make("doapp", settings, wrappers_settings)

    obs = env.reset()

    while True:

        env.render()

        action = agent.compute_single_action(observation=obs, explore=True, policy_id="default_policy")

        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()
            if info["env_done"]:
                break

    # Close the environment
    env.close()