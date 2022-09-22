import sys
import os
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, ".."))
import diambra.arena
from make_ray_env import DiambraArena
from ray.rllib.algorithms.ppo import PPO

if __name__ == "__main__":

    # Settings
    settings = {}
    settings["hardcore"] = True
    settings["frame_shape"] = [84, 84, 1]
    settings["characters"] = [["Kasumi"], ["Kasumi"]]

    # Wrappers Settings
    wrappers_settings = {}
    #wrappers_settings["reward_normalization"] = True
    #wrappers_settings["frame_stack"] = 5

    config = {
        # Define and configure the environment
        "env": DiambraArena,
        "env_config": {
            "game_id": "doapp",
            "settings": settings,
            "wrappers_settings": wrappers_settings
        },
        # Configure the algorithm
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        #"model": {
        #    "custom_model": "my_model",
        #    "vf_share_layers": True,
        #},
        "num_workers": diambra.arena.get_num_envs(),  # parallelism
        "framework": "torch",

        # Set up a separate evaluation worker set for the
        # `algo.evaluate()` call after training (see below).
        "evaluation_num_workers": 1,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
        }
    }

    # Create our RLlib Trainer.
    algo = PPO(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    print("\nStarting training ...\n")
    for idx in range(3):
        print("Training iteration:", idx + 1)
        print(algo.train())
    print("\n .. training completed.")

    # Evaluate the trained Trainer (and render each timestep to the shell's
    # output).
    print("\nStarting evaluation ...\n")
    algo.evaluate()
    print("\n... evaluation completed.\n")