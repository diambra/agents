import diambra.arena
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    # Create environment
    env = diambra.arena.make("doapp", render_mode="human")

    # Instantiate the agent
    agent = A2C("MultiInputPolicy", env, verbose=1)
    # Train the agent
    agent.learn(total_timesteps=200)
    # Save the agent
    agent.save("a2c_doapp")
    del agent  # delete trained agent to demonstrate loading

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the agent was trained vs the current one
    # agent = A2C.load("a2c_doapp", env=env, print_system_info=True)
    agent = A2C.load("a2c_doapp", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=3)
    print("Reward: {} (avg) ± {} (std)".format(mean_reward, std_reward))

    # Run trained agent
    observation, info = env.reset()
    cumulative_reward = 0
    while True:
        env.render()

        action, _state = agent.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        if (reward != 0):
            print("Cumulative reward =", cumulative_reward)

        if terminated or truncated:
            observation, info = env.reset()
            break

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    main()