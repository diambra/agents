import sys
import os
import diambra.arena
from make_ray_env import DiambraArena, preprocess_ray_config
from ray.rllib.algorithms.ppo import PPO

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

        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
        },
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
    print("Training results: {}".format(results))

    '''
    # Save the agent
    model.save("a2c_doapp")
    del model  # delete trained model to demonstrate loading

    # Load the trained agent
    model = A2C.load("a2c_doapp", env=env)
    '''

    # Evaluate the trained agent (and render each timestep to the shell's
    # output).
    print("\nStarting evaluation ...\n")
    results = agent.evaluate()
    print("\n... evaluation completed.\n")
    print("Evaluation results: {}".format(results))