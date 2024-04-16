import argparse
import json

import gymnasium as gym
import torch
from lightning import Fabric
from omegaconf import OmegaConf
from sheeprl.algos.ppo.agent import build_agent
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict

"""This is an example agent based on SheepRL.

Usage:
cd sheeprl
diambra run python ppo-agent.py --cfg_path "./fake-logs/runs/ppo/doapp/fake-experiment/version_0/config.yaml" --checkpoint_path "./fake-logs/runs/ppo/doapp/fake-experiment/version_0/checkpoint/ckpt_1024_0.ckpt"
"""


def main(cfg_path: str, checkpoint_path: str, test=False):
    # Read the cfg file
    cfg = dotdict(OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True))
    print("Config parameters = ", json.dumps(cfg, sort_keys=True, indent=4))

    # Override configs for evaluation
    # You do not need to capture the video since you are submitting the agent and the video is recorded by DIAMBRA
    cfg.env.capture_video = False

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
    mlp_keys = cfg.algo.mlp_keys.encoder
    obs_keys = mlp_keys + cnn_keys

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
        agent_state=state["agent"],
    )[-1]
    agent.eval()

    # Print policy network architecture
    print("Policy architecture:")
    print(agent)

    o, info = env.reset()

    while True:
        # Convert numpy observations into torch observations and normalize image observations
        # Every algorithm has its own way to do it, check in the test function of the algorithm
        # which is the correct way to it.
        # Check the `test()` function called in the `evaluate.py` file of the algorithm.
        obs = {}
        for k in o.keys():
            if k in obs_keys:
                torch_obs = torch.from_numpy(o[k].copy()).to(fabric.device).unsqueeze(0)
                if k in cnn_keys:
                    torch_obs = (
                        torch_obs.reshape(1, -1, *torch_obs.shape[-2:]) / 255 - 0.5
                    )
                if k in mlp_keys:
                    torch_obs = torch_obs.float()
                obs[k] = torch_obs

        # Select actions, the agent returns a one-hot categorical or
        # more one-hot categorical distributions for muli-discrete actions space
        actions = agent.get_actions(obs, greedy=True)
        # Convert actions from one-hot categorical to categorial
        actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1)

        o, _, terminated, truncated, info = env.step(
            actions.cpu().numpy().reshape(env.action_space.shape)
        )

        if terminated or truncated:
            o, info = env.reset()
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
