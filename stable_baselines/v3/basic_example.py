import diambra.arena
from stable_baselines3 import A2C

env = diambra.arena.make("doapp", {"hardcore": True, "frame_shape": [128, 128, 1]})

model = A2C('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=1000)

observation = env.reset()
while True:
    env.render()

    action, _state = model.predict(observation, deterministic=True)

    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
        break

env.close()
