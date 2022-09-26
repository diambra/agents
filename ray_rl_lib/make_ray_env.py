import os
import sys
import diambra.arena
import logging
import gym
from ray.rllib.env.env_context import EnvContext
from copy import deepcopy

class DiambraArena(gym.Env):

    def __init__(self, config: EnvContext):

        logger = logging.getLogger(__name__)

        if "is_rollout" not in config.keys():
            message = "Environment initialized without a preprocessed config file."
            message += " Make sure to call \"preprocess_ray_config\" before inizializing Ray RL Algorithms."
            raise Exception(message)

        self.game_id = config["game_id"]
        self.settings = config["settings"] if "settings" in config.keys() else {}
        self.wrappers_settings = config["wrappers_settings"] if "wrappers_settings" in config.keys() else {}
        self.seed = config["seed"] if "seed" in config.keys() else 0

        num_rollout_workers = config["num_workers"]
        num_eval_workers = config["evaluation_num_workers"]
        num_envs_per_worker = config["num_envs_per_worker"]
        worker_index = config.worker_index
        vector_index = config.vector_index
        create_env_on_driver = config["create_env_on_driver"]
        is_rollout = config["is_rollout"]

        if is_rollout is True:
            self.rank = vector_index + (worker_index - 1) * num_envs_per_worker
        else:
            self.rank = vector_index + (worker_index - 1) * num_envs_per_worker + num_rollout_workers * num_envs_per_worker

        if create_env_on_driver is True:
            self.rank += num_envs_per_worker

        print("RANK = ", self.rank)

        self.env = diambra.arena.make(self.game_id, self.settings, self.wrappers_settings,
                                      seed=self.seed + self.rank, rank=self.rank)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

def preprocess_ray_config(config):

    logger = logging.getLogger(__name__)

    num_envs_required = 0
    num_envs_per_worker = config["num_envs_per_worker"] if "num_envs_per_worker" in config.keys() else 1

    # Set create_env_on_driver appropriately
    if ("num_workers" not in config.keys() or config["num_workers"] == 0) or\
       (("evaluation_interval" in config.keys() and config["evaluation_interval"] is not None and config["evaluation_interval"] > 0) and\
        ("evaluation_num_workers" not in config.keys() or ("evaluation_num_workers" in config.keys() and config["evaluation_num_workers"] == 0))) :
        config["create_env_on_driver"] = True

    # Count an additional env if create_env_on_driver is set as True
    if config["create_env_on_driver"] is True:
        num_envs_required += num_envs_per_worker

    # Define rollout workers related variables
    num_workers = config["num_workers"] if "num_workers" in config.keys() else 0
    # Add rollout workers
    num_envs_required += num_workers*num_envs_per_worker

    # Define evaluation workers related variables
    evaluation_num_workers = config["evaluation_num_workers"] if "evaluation_num_workers" in config.keys() else 0
    # Add evaluation workers
    num_envs_required += evaluation_num_workers*num_envs_per_worker

    #logger.debug("Total number of environments required = {}".format(num_envs_required))
    print("Total number of environments required = {}".format(num_envs_required))

    if num_envs_required > diambra.arena.get_num_envs():
        message = "Config file requires more environments than"
        message += " those instantiated by the CLI: {} VS {}.".format(num_envs_required, diambra.arena.get_num_envs())
        message += " Use the CLI option '-s={}' for this configuration.".format(num_envs_required)
        raise Exception(message)
    elif num_envs_required > diambra.arena.get_num_envs():
        message = "Config file requires less environments than"
        message += " those instantiated by the CLI: {} VS {}.".format(num_envs_required, diambra.arena.get_num_envs())
        message += " To avoid wasting resources, use the CLI option '-s={}' for this configuration.".format(num_envs_required)
        logger.warning(message)

    config["env_config"]["create_env_on_driver"] = config["create_env_on_driver"] if "create_env_on_driver" in config.keys() else False

    # Setting flag to let the env class know if it is an evaluation env or not
    config["env_config"]["is_rollout"] = True
    config["env_config"]["num_workers"] = num_workers
    config["env_config"]["evaluation_num_workers"] = evaluation_num_workers
    config["env_config"]["num_envs_per_worker"] = num_envs_per_worker

    if "evaluation_config" not in config.keys():
        config["evaluation_config"] = {}
    if "env_config" not in config["evaluation_config"].keys():
        config["evaluation_config"]["env_config"] = deepcopy(config["env_config"])

    config["evaluation_config"]["env_config"]["is_rollout"] = False

    return config

'''
# === Settings for Rollout Worker processes ===
# Number of rollout worker actors to create for parallel sampling. Setting
# this to 0 will force rollouts to be done in the algorithm's actor.
"num_workers": 2,
# Number of environments to evaluate vector-wise per worker. This enables
# model inference batching, which can improve performance for inference
# bottlenecked workloads.
"num_envs_per_worker": 1,
# When `num_workers` > 0, the driver (local_worker; worker-idx=0) does not
# need an environment. This is because it doesn't have to sample (done by
# remote_workers; worker_indices > 0) nor evaluate (done by evaluation
# workers; see below).
"create_env_on_driver": False,

# Whether to use "rllib" or "deepmind" preprocessors by default
# Set to None for using no preprocessor. In this case, the model will have
# to handle possibly complex observations from the environment.
"preprocessor_pref": "deepmind",

# === Evaluation Settings ===
# Evaluate with every `evaluation_interval` training iterations.
# The evaluation stats will be reported under the "evaluation" metric key.
# Note that for Ape-X metrics are already only reported for the lowest
# epsilon workers (least random workers).
# Set to None (or 0) for no evaluation.
"evaluation_interval": None,
# Number of parallel workers to use for evaluation. Note that this is set
# to zero by default, which means evaluation will be run in the algorithm
# process (only if evaluation_interval is not None). If you increase this,
# it will increase the Ray resource usage of the algorithm since evaluation
# workers are created separately from rollout workers (used to sample data
# for training).
"evaluation_num_workers": 0,
'''
