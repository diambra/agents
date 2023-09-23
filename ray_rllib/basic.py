import diambra.arena
from diambra.arena import SpaceTypes, EnvironmentSettings
import gymnasium as gym
from diambra.arena.ray_rllib.make_ray_env import DiambraArena, preprocess_ray_config
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import pretty_print

def main():
    # Environment Settings
    env_settings = EnvironmentSettings()
    env_settings.frame_shape = (84, 84, 1)
    env_settings.action_space = SpaceTypes.DISCRETE

    # env_config
    env_config = {
            "game_id": "doapp",
            "settings": env_settings,
        }

    config = {
        # Define and configure the environment
        "env": DiambraArena,
        "env_config": env_config,
        "num_workers": 0,
        "train_batch_size": 200,
    }

    # Update config file
    config = preprocess_ray_config(config)

    # Instantiating the agent
    agent = PPO(config=config)

    # Run it for n training iterations
    print("\nStarting training ...\n")
    for idx in range(1):
        print("Training iteration:", idx + 1)
        result = agent.train()
        print(pretty_print(result))
    print("\n .. training completed.")

    # Run the trained agent (and render each timestep output).
    print("\nStarting trained agent execution ...\n")

    env = diambra.arena.make("doapp", env_settings, render_mode="human")

    observation, info = env.reset()
    while True:
        env.render()

        action = agent.compute_single_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            break

    print("\n... trained agent execution completed.\n")

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    main()