#!/usr/bin/env python3
import diambra.arena
from diambra.arena.utils.gym_utils import env_spaces_summary, available_games
import random
import argparse

def main(game_id="random"):

    game_dict = available_games(False)
    if game_id == "random":
        game_id = random.sample(game_dict.keys(),1)[0]
    else:
        game_id = opt.gameId if opt.gameId in game_dict.keys() else random.sample(game_dict.keys(),1)[0]

    # Settings
    settings = {
        "player": "P2",
        "step_ratio": 2,
        "frame_shape": (256, 256, 0),
        "hardcore": False,
        "difficulty": 4,
        "characters": ("Random"),
        "char_outfits": 1,
        "action_space": "discrete",
        "attack_but_combination": True
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

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gameId', type=str, default="random", help='Game ID')
    opt = parser.parse_args()
    print(opt)

    main(opt.gameId)
