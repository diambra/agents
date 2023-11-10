# Diambra Agents

import hydra
from diambra.arena.sheeprl import CONFIGS_PATH
from omegaconf import DictConfig

from sheeprl.cli import evaluation


@hydra.main(version_base="1.3", config_path=CONFIGS_PATH, config_name="eval_config")
def run(cfg: DictConfig):
    evaluation(cfg)


if __name__ == "__main__":
    run()
