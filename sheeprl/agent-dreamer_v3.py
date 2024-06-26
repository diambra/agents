import argparse
import json

import gymnasium as gym
import torch
from lightning import Fabric
from omegaconf import OmegaConf
from sheeprl.algos.dreamer_v3.agent import build_agent
from sheeprl.algos.dreamer_v3.utils import prepare_obs
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict

"""This is an example agent based on SheepRL.

Usage:
cd sheeprl
diambra run python agent-dreamer_v3.py --cfg_path "/absolute/path/to/example-logs/runs/dreamer_v3/doapp/experiment/version_0/config.yaml" --checkpoint_path "/absolute/path/to/example-logs/runs/dreamer_v3/doapp/experiment/version_0/checkpoint/ckpt_1024_0.ckpt"
"""


def main(cfg_path: str, checkpoint_path: str, test=False):
    # Read the cfg file
    cfg = dotdict(OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True))
    print("Config parameters = ", json.dumps(cfg, sort_keys=True, indent=4))

    # Override configs for evaluation
    # You do not need to capture the video since you are submitting the agent and the video is recorded by DIAMBRA
    cfg.env.capture_video = False
    # Only one environment is used for evaluation
    cfg.env.num_envs = 1

    # Instantiate Fabric
    # You must use the same precision and plugins used for training.
    precision = getattr(cfg.fabric, "precision", None)
    plugins = getattr(cfg.fabric, "plugins", None)
    fabric = Fabric(
        accelerator="auto",
        devices=1,
        num_nodes=1,
        precision=precision,
        plugins=plugins,
        strategy="auto",
    )

    # Create Environment
    env = make_env(cfg, 0, 0)()
    observation_space = env.observation_space
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n]
    )
    cnn_keys = cfg.algo.cnn_keys.encoder

    # Load the trained agent
    state = fabric.load(checkpoint_path)
    # You need to retrieve only the player
    # Check for each algorithm what models the `build_agent()` function returns
    # (placed in the `agent.py` file of the algorithm), and which arguments it needs.
    # Check also which are the keys of the checkpoint: if the `build_agent()` parameter
    # is called `model_state`, then you retrieve the model state with `state["model"]`.
    agent = build_agent(
        fabric=fabric,
        actions_dim=actions_dim,
        is_continuous=False,
        cfg=cfg,
        obs_space=observation_space,
        world_model_state=state["world_model"],
        actor_state=state["actor"],
        critic_state=state["critic"],
        target_critic_state=state["target_critic"],
    )[-1]
    agent.eval()

    # Print policy network architecture
    print("Policy architecture:")
    print(agent)

    obs, info = env.reset()
    # Every time you reset the environment, you must reset the initial states of the model
    agent.init_states()

    while True:
        # Convert numpy observations into torch observations and normalize image observations
        # Every algorithm has its own way to do it, you must import the correct method
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cnn_keys)

        # Select actions, the agent returns a one-hot categorical or
        # more one-hot categorical distributions for muli-discrete actions space
        actions = agent.get_actions(torch_obs, greedy=False)
        # Convert actions from one-hot categorical to categorial
        actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1)

        obs, _, terminated, truncated, info = env.step(
            actions.cpu().numpy().reshape(env.action_space.shape)
        )

        if terminated or truncated:
            obs, info = env.reset()
            # Every time you reset the environment, you must reset the initial states of the model
            agent.init_states()
            if info["env_done"] or test is True:
                break

    # Close the environment
    env.close()

    # Return success
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path", type=str, required=True, help="Configuration file"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="model", help="Model checkpoint"
    )
    parser.add_argument("--test", action="store_true", help="Test mode")
    opt = parser.parse_args()
    print(opt)

    main(opt.cfg_path, opt.checkpoint_path, opt.test)
