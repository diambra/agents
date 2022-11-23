#!/usr/bin/env python3
import diambra.arena
from diambra.arena.utils.gym_utils import env_spaces_summary

if __name__ == "__main__":

    # Settings
    settings = {
        "player": "P2",
        "step_ratio": 2,
        "frame_shape": (256, 256, 3),
        "hardcore": False,
        "difficulty": 4,
        "characters": ("Random"),
        "char_outfits": 3,
        "action_space": "discrete",
        "attack_but_combination": True
    }

    env = diambra.arena.make("doapp", settings)

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
