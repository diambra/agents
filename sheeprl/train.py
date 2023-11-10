# Diambra Agents

import hydra
from diambra.arena.sheeprl import CONFIGS_PATH
from omegaconf import DictConfig

from sheeprl.cli import run


def check_configs(cfg: DictConfig):
    if "diambra" not in cfg.env.wrapper._target_:
        raise ValueError(
            f"You must choose a DIAMBRA environment. "
            f"Got '{cfg.env.id}' provided by '{cfg.env.wrapper._target_.split('.')[-2]}'."
        )


@hydra.main(version_base="1.3", config_path=CONFIGS_PATH, config_name="config")
def train(cfg: DictConfig):
    check_configs(cfg)
    run(cfg)


if __name__ == "__main__":
    train()
