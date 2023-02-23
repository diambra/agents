#!/usr/bin/env python3
import diambra.arena
from diambra.arena.utils.gym_utils import env_spaces_summary, available_games
import random
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gameId', type=str, default="random", help='Game ID')
    opt = parser.parse_args()
    print(opt)

    game_dict = available_games(False)
    if opt.gameId == "random":
        game_id = random.sample(game_dict.keys(),1)[0]
    else:
        game_id = opt.gameId if opt.gameId in game_dict.keys() else random.sample(game_dict.keys(),1)[0]

    # Settings
    settings = {
        "player": "P2",
        "step_ratio": 6,
        "frame_shape": (128, 128, 1),
        "hardcore": False,
        "difficulty": 4,
        "characters": ("Random"),
        "char_outfits": 1,
        "action_space": "multi_discrete",
        "attack_but_combination": False
    }

    env = diambra.arena.make(game_id, settings)

    env_spaces_summary(env)

    observation = env.reset()
    env.show_obs(observation)

    while True:

        actions = env.action_space.sample()

        observation, reward, done, info = env.step(actions)
        env.show_obs(observation)
        print("Reward: {}".format(reward))
        print("Done: {}".format(done))
        print("Info: {}".format(info))

        if done:
            observation = env.reset()
            env.show_obs(observation)
            if info["env_done"]:
                break

    env.close()
