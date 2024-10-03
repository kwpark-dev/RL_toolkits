import numpy as np
import gymnasium as gym


env = gym.make('MountainCarContinuous-v0', render_mode='human')
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(np.array([5.]))

    print(action, observation, reward, terminated, truncated, info)
    
    if terminated or truncated:
        observation, info = env.reset()

        env.close()

