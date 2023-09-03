#!/usr/bin/env python3
import diambra.arena
from diambra.arena.utils.gym_utils import env_spaces_summary, available_games
import random
import argparse

def main(game_id="random", test=False):
    game_dict = available_games(False)
    if game_id == "random":
        game_id = random.sample(game_dict.keys(),1)[0]
    else:
        game_id = opt.gameId if opt.gameId in game_dict.keys() else random.sample(game_dict.keys(),1)[0]

    # Settings
    settings = {
        "n_players": 1,
        "step_ratio": 2,
        "frame_shape": (256, 256, 0),
        "role": "P2",
        "difficulty": 4,
        "characters": ("Random"),
        "outfits": 1,
        "action_space": "discrete",
    }

    env = diambra.arena.make(game_id, settings)
    env_spaces_summary(env)

    observation, info = env.reset()
    print("Info: {}".format(info))
    env.show_obs(observation)

    while True:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        env.show_obs(observation)
        print("Reward: {}".format(reward))
        print("Terminated: {}".format(terminated))
        print("Truncated: {}".format(truncated))

        if terminated or truncated:
            observation, info = env.reset()
            print("Info: {}".format(info))
            env.show_obs(observation)
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
