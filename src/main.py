import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="human")

print(env.action_space)
print(env.observation_space)

observation, info = env.reset(seed=42)
for i in range(1000):
    # action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(i % 2)
    print(observation, reward, terminated, truncated, info)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
