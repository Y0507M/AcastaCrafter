import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from craftium import CraftiumEnv
import os
import time
import craftium
import sys


# Create output directory for images
if not os.path.exists('observations'):
    os.makedirs('observations')

try:
    # Create the environment
    print("Creating environment...")
    env = CraftiumEnv(
    env_dir="craftium-envs/chop-tree",
    render_mode="human",
    obs_width=512,
    obs_height=512,
  )
    print("Environment created successfully")
    
    # Add a small delay to allow Minetest to initialize
    time.sleep(2)
    
    print("Attempting initial reset...")
    observation, info = env.reset()
    print("Reset successful")
    
    # Rest of your code...
    for t in range(10000):
        action = env.action_space.sample()
        if t % 200 == 0:
            plt.imsave(f'observations/step_{t}.png', np.transpose(observation, (1, 0, 2)))
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {t}: Action {action}, Reward {reward}, Terminated {terminated}, Truncated {truncated}")
        env.render()
        if terminated or truncated:
            observation, info = env.reset()

except Exception as e:
    print(f"Error occurred: {e}")
    print(f"Error type: {type(e)}")
    print(f"Python version: {sys.version}")
    print(f"Craftium version: {craftium.__version__ if hasattr(craftium, '__version__') else 'Unknown'}")
finally:
    try:
        env.close()
    except:
        pass
