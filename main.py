import gymnasium as gym





# Create the environment without specifying the render mode
env = gym.make("HalfCheetah-v5", render_mode='human', width=800, height=800)

# Reset the environment to start
observation, info = env.reset()

# Render and run the environment using mujoco-py
for _ in range(1000):
    action = env.action_space.sample()  # Sample random actions
    observation, reward, done, truncated, info = env.step(action)

    if done or truncated:
        observation, info = env.reset()

env.close()
