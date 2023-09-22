#!/usr/bin/env python3
import diambra.arena
from diambra.arena import SpaceTypes, Roles, EnvironmentSettings
from diambra.arena.utils.gym_utils import available_games
import random
import argparse

def main(game_id="random", test=False):
    game_dict = available_games(False)
    if game_id == "random":
        game_id = random.sample(game_dict.keys(),1)[0]
    else:
        game_id = opt.gameId if opt.gameId in game_dict.keys() else random.sample(game_dict.keys(),1)[0]

    # Settings
    settings = EnvironmentSettings()
    settings.step_ratio = 6
    settings.frame_shape = (128, 128, 1)
    settings.role = Roles.P2
    settings.difficulty = 4
    settings.action_space = SpaceTypes.MULTI_DISCRETE

    env = diambra.arena.make(game_id, settings)
    observation, info = env.reset()

    while True:
        action = env.get_no_op_action()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            if info["env_done"] or test is True:
                break

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gameId', type=str, default="random", help='Game ID')
    parser.add_argument('--test', type=int, default=0, help='Test mode')
    opt = parser.parse_args()
    print(opt)

    main(opt.gameId, bool(opt.test))
