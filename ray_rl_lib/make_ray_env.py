import os
import sys
import diambra.arena
import ray
import gym
from ray.rllib.env.env_context import EnvContext

from ray.rllib.algorithms.ppo import PPO

class DiambraArena(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.game_id = config["game_id"]
        self.settings = config["settings"] if "settings" in config.keys() else {}
        self.wrappers_settings = config["wrappers_settings"] if "wrappers_settings" in config.keys() else None
        self.seed = config["seed"] if "seed" in config.keys() else 0
        self.rank = config.worker_index - 1

        env_addresses = os.getenv("DIAMBRA_ENVS", "").split()
        if len(env_addresses) == 0:
            raise Exception("ERROR: Running script without DIAMBRA CLI.")
            sys.exit(1)
        elif len(env_addresses) != config.num_workers:
            raise Exception("ERROR: Number of rollout workers different than the number of activated environments via DIAMBRA CLI.")
            sys.exit(1)

        self.env = diambra.arena.make(self.game_id, self.settings, self.wrappers_settings,
                                      seed=self.seed + self.rank, rank=self.rank)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        print("Initialized Env with rank {} of {}".format(config.worker_index, config.num_workers))

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

if __name__ == "__main__":

    ray.init(local_mode=True)

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
        "env": DiambraArena,
        "env_config": {
            "game_id": "doapp",
            "settings": settings,
            "wrappers_settings": wrappers_settings,
            "seed": 0
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        #"model": {
        #    "custom_model": "my_model",
        #    "vf_share_layers": True,
        #},
        "num_workers": diambra.arena.get_num_envs(),  # parallelism
        "framework": "torch",
    }

    # Create our RLlib Trainer.
    algo = PPO(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for idx in range(3):
        print("Training iteration:", idx)
        print(algo.train())