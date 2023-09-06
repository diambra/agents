import diambra.arena
from stable_baselines3 import A2C

def main():
    env = diambra.arena.make("doapp", render_mode="human")

    print("\nStarting training ...\n")
    agent = A2C("MultiInputPolicy", env, verbose=1)
    agent.learn(total_timesteps=200)
    print("\n .. training completed.")

    print("\nStarting trained agent execution ...\n")
    observation, info = env.reset()
    while True:
        env.render()

        action, _state = agent.predict(observation, deterministic=True)

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
