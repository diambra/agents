import diambra.arena
from stable_baselines3 import A2C

if __name__ == "__main__":

    env = diambra.arena.make("doapp", {"hardcore": True, "frame_shape": [128, 128, 1]})

    print("\nStarting training ...\n")
    agent = A2C('CnnPolicy', env, verbose=1)
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

    env.close()
