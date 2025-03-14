import gym
import minerl
import numpy as np

from stable_baselines import PPO
from stable_baselines.common.evaluation import evaluate_policy

from wrappers import ObservationShaping, ActionShaping


N_EPISODES = 10

env = gym.make("MineRLTreechop-v0")
env.make_interactive(port=None, realtime=False)
env = ObservationShaping(env)
env = ActionShaping(env)

model_paths = ['model_treechop_2025-03-13_20-55-53.zip',
               'model_treechop_2025-03-14_00-27-48.zip',
               'model_treechop_2025-03-14_07-01-52.zip']

# baseline
def evaluate_baseline():
    episode_rewards, episode_lengths = [], []
    for i in range(N_EPISODES):
        env.reset()

        done = False
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    print(f'Baseline mean reward: {np.mean(episode_rewards)}')
    print(f'Baseline std reward: {np.std(episode_rewards)}')

#evaluate_baseline()

for model_path in model_paths:
    model = PPO2.load(model_path)
    mean_reward, std_reward = evaluate_policy(model, env, N_EPISODES)
    print(f'Model {model_path} mean reward: {mean_reward}')
    print(f'Model {model_path} std reward: {std_reward}')

