import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import craftium
from craftium import CraftiumEnv

env = CraftiumEnv(
    env_dir="craftium-envs/chop-tree",
    render_mode="human",
    obs_width=512,
    obs_height=512,
)

observation, info = env.reset()

for t in range(1000):
    action = env.action_space.sample()

    # plot the observation
    plt.clf()
    plt.imshow(np.transpose(observation, (1, 0, 2)))
    plt.pause(0.02)  # wait for 0.02 seconds

    observation, reward, terminated, truncated, _info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()