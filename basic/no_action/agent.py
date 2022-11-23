#!/usr/bin/env python3
import diambra.arena
from diambra.arena.utils.gym_utils import env_spaces_summary

if __name__ == "__main__":

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

    env = diambra.arena.make("doapp", settings)

    env_spaces_summary(env)

    observation = env.reset()
    env.show_obs(observation)

    while True:

        action = 0 if settings["action_space"] == "discrete" else [0, 0]

        observation, reward, done, info = env.step(action)
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
