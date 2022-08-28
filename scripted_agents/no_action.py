#!/usr/bin/env python3
import diambra.arena
from diambra.arena.utils.gym_utils import show_gym_obs, env_spaces_summary

# Settings
settings = {
    "player": "P2",
    "step_ratio": 6,
    "frame_shape": [128, 128, 1],
    "hardcore": False,
    "difficulty": 4,
    "characters": [["Random"], ["Random"]],
    "char_outfits": [1, 1],
    "action_space": "multi_discrete",
    "attack_but_combination": False
}

env = diambra.arena.make("doapp", settings)

env_spaces_summary(env)

observation = env.reset()
show_gym_obs(observation, env.char_names)

while True:

    actions = env.action_space.sample()

    observation, reward, done, info = env.step(actions)
    show_gym_obs(observation, env.char_names)
    print("Reward: {}".format(reward))
    print("Done: {}".format(done))
    print("Info: {}".format(info))

    if done:
        observation = env.reset()
        show_gym_obs(observation, env.char_names)

env.close()
