import os
import sys
import diambra.arena
import gym
from ray.rllib.env.env_context import EnvContext

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

        print("Initialized Env with (0-based) rank {} of {}".format(self.rank, config.num_workers))

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)