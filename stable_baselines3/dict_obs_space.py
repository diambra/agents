from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env
from stable_baselines3 import PPO

def main():
    # Settings
    settings = {}
    settings["frame_shape"] = (128, 128, 1)
    settings["characters"] = ("Kasumi")

    # Wrappers Settings
    wrappers_settings = {}
    wrappers_settings["reward_normalization"] = True
    wrappers_settings["actions_stack"] = 12
    wrappers_settings["frame_stack"] = 5
    wrappers_settings["scale"] = True
    wrappers_settings["exclude_image_scaling"] = True
    wrappers_settings["flatten"] = True
    wrappers_settings["filter_keys"] = ["stage", "P1_ownHealth", "P1_oppHealth", "P1_ownSide",
                                        "P1_oppSide", "P1_oppChar", "P1_actions_move", "P1_actions_attack"]

    # Create environment
    env, num_envs = make_sb3_env("doapp", settings, wrappers_settings)
    print("Activated {} environment(s)".format(num_envs))

    print("Observation space =", env.observation_space)
    print("Act_space =", env.action_space)

    # Instantiate the agent
    agent = PPO("MultiInputPolicy", env, verbose=1)

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Train the agent
    agent.learn(total_timesteps=200)

    # Run trained agent
    observation = env.reset()
    cumulative_reward = [0.0 for _ in range(num_envs)]
    while True:
        env.render()

        action, _state = agent.predict(observation, deterministic=True)

        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if any(x != 0 for x in reward):
            print("Cumulative reward(s) =", cumulative_reward)

        if done.any():
            observation = env.reset()
            break

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    main()
