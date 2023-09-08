import argparse
import diambra.arena
from diambra.arena.ray_rllib.make_ray_env import DiambraArena, preprocess_ray_config
from ray.rllib.algorithms.ppo import PPO

# Reference: https://github.com/ray-project/ray/blob/ray-2.0.0/rllib/examples/inference_and_serving/policy_inference_after_training.py

"""This is an example agent based on RL Lib.

Usage:
diambra run python agent.py --trainedModel /absolute/path/to/checkpoint/ --envSpaces /absolute/path/to/environment/spaces/descriptor/
"""

def main(trained_model, env_spaces, test=False):
    # Settings
    env_settings = {}
    env_settings["frame_shape"] = (84, 84, 1)
    env_settings["characters"] = ("Kasumi")
    env_settings["action_space"] = "discrete"

    # Wrappers Settings
    wrappers_settings = {}
    wrappers_settings["reward_normalization"] = True
    wrappers_settings["actions_stack"] = 12
    wrappers_settings["frame_stack"] = 5
    wrappers_settings["scale"] = True

    config = {
        # Define and configure the environment
        "env": DiambraArena,
        "env_config": {
            "game_id": "doapp",
            "settings": env_settings,
            "wrappers_settings": wrappers_settings,
            "load_spaces_from_file": True,
            "env_spaces_file_name": env_spaces,
        },
        "num_workers": 0,
    }

    # Update config file
    config = preprocess_ray_config(config)

    # Load the trained agent
    agent = PPO(config=config)
    agent.restore(trained_model)
    print("Agent loaded")

    # Print the agent policy architecture
    print("Policy architecture =\n{}".format(agent.get_policy().model))

    env = diambra.arena.make("doapp", env_settings, wrappers_settings, render_mode="human")

    obs, info = env.reset()

    while True:
        env.render()

        action = agent.compute_single_action(observation=obs, explore=True, policy_id="default_policy")

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            if info["env_done"] or test is True:
                break

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainedModel", type=str, required=True, help="Model path")
    parser.add_argument("--envSpaces", type=str, required=True, help="Environment spaces descriptor file path")
    parser.add_argument("--test", type=int, default=0, help="Test mode")
    opt = parser.parse_args()
    print(opt)

    main(opt.trainedModel, opt.envSpaces, bool(opt.test))