import diambra.arena
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = diambra.arena.make("doapp", {"hardcore": True, "frame_shape": [128, 128, 1]})

# Instantiate the agent
model = A2C('CnnPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=1000)
# Save the agent
model.save("a2c_doapp")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = A2C.load("a2c_doapp", env=env, print_system_info=True)
model = A2C.load("a2c_doapp", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=3)
print("Reward: {} (avg) ± {} (std)".format(mean_reward, std_reward))

# Enjoy trained agent
observation = env.reset()
cumulative_reward = 0
while True:
    env.render()

    action, _state = model.predict(observation, deterministic=True)

    observation, reward, done, info = env.step(action)
    cumulative_reward += reward
    if (reward != 0):
        print("Cumulative reward =", cumulative_reward)

    if done:
        observation = env.reset()
        break

env.close()
