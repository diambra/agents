from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import A2C

def main():
    # Settings
    settings = EnvironmentSettings()
    settings.frame_shape = (128, 128, 1)
    settings.characters = ("Kasumi")

    # Wrappers Settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.reward_normalization = True
    wrappers_settings.frame_stack = 5
    wrappers_settings.add_last_action_to_observation = True
    wrappers_settings.actions_stack = 12
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.role_relative_observation = True
    wrappers_settings.flatten = True
    wrappers_settings.filter_keys = ["action", "own_health", "opp_health", "own_side", "opp_side", "opp_character", "stage", "timer"]

    # Create environment
    env, num_envs = make_sb3_env("doapp", settings, wrappers_settings)
    print("Activated {} environment(s)".format(num_envs))

    print("\nStarting training ...\n")
    agent = A2C("MultiInputPolicy", env, verbose=1)
    agent.learn(total_timesteps=200)
    print("\n .. training completed.")

    print("\nStarting trained agent execution ...\n")
    observation = env.reset()
    while True:
        env.render()

        action, _state = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()
            break
    print("\n... trained agent execution completed.\n")

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    main()
