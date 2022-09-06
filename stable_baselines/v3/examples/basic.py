import diambra.arena
from stable_baselines3 import A2C

env = diambra.arena.make("doapp", {"hardcore": True, "frame_shape": [128, 128, 1]})

print("\nStarting training ...\n")
model = A2C('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=1000)
print("\n .. training completed.")

print("\nStarting evaluation ...\n")
observation = env.reset()
while True:
    env.render()

    action, _state = model.predict(observation, deterministic=True)

    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
        break
print("\n... evaluation completed.\n")

env.close()
