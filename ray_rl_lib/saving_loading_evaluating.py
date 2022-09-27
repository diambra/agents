import sys
import os
import diambra.arena
from make_ray_env import DiambraArena, preprocess_ray_config
from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import pretty_print

if __name__ == "__main__":

    # Settings
    settings = {}
    settings["hardcore"] = True
    settings["frame_shape"] = [84, 84, 1]

    config = {
        # Define and configure the environment
        "env": DiambraArena,
        "env_config": {
            "game_id": "doapp",
            "settings": settings,
        },
        "num_workers": 0,
        "train_batch_size": 200,
    }

    # Update config file
    config = preprocess_ray_config(config)

    # Create the RLlib Agent.
    agent = PPO(config=config)

    # Run it for n training iterations
    print("\nStarting training ...\n")
    for idx in range(1):
        print("Training iteration:", idx + 1)
        results = agent.train()
    print("\n .. training completed.")
    print("Training results:\n{}".format(pretty_print(results)))

    # Save the agent
    checkpoint = agent.save()
    print("Checkpoint saved at {}".format(checkpoint))
    del agent  # delete trained model to demonstrate loading

    # Load the trained agent
    agent = PPO(config=config)
    agent.restore(checkpoint)
    print("Agent loaded")

    # Evaluate the trained agent (and render each timestep to the shell's
    # output).
    print("\nStarting evaluation ...\n")
    results = agent.evaluate()
    print("\n... evaluation completed.\n")
    print("Evaluation results:\n{}".format(pretty_print(results)))