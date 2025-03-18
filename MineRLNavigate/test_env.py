import gym
import minerl
import os

# Set environment variables
os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = './minecraft_logs'
os.environ['MALMO_XMS'] = '2G'
os.environ['MALMO_XMX'] = '4G'
os.environ['MALMO_MINECRAFT_PORT'] = '10000'
os.environ['MINERL_HEADLESS'] = '1'

# Create and test the environment
env = gym.make("MineRLNavigate-v0")
obs = env.reset()
print("Environment reset successfully!")
print(f"Observation shape: {obs['pov'].shape}")

# Take a random action
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
print(f"Step taken, reward: {reward}")

env.close()