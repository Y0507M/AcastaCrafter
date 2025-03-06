import gym
import minerl
import numpy as np
import cv2
from datetime import datetime
import os

from gym import ObservationWrapper, ActionWrapper, RewardWrapper
from gym.spaces import Box, Discrete
from gym.wrappers.time_limit import TimeLimit

import torch as th
import torch.nn as nn

device = 'cuda' if th.cuda.is_available() else 'cpu'

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from tensorboard_video_recorder import TensorboardVideoRecorder

# class ChoptreeObservationWrapper(ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.observation_space = Box(low=0, high=255, shape=(3, 360, 640), dtype=np.uint8)  # Full RGB, larger resolution

#     def observation(self, obs):
#         pov = obs['pov']  # Original environment RGB image (likely 160x120 or similar)
        
#         # Resize while keeping the 3-channel RGB format
#         resized_pov = cv2.resize(pov, (640, 360), interpolation=cv2.INTER_LINEAR)  # Now (360, 640, 3)

#         # Convert from (H, W, C) → (C, H, W) for PyTorch compatibility
#         return np.transpose(resized_pov, (2, 0, 1))


class ChoptreeActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._actions = [
            {'forward': 1},  # Move forward
            {'jump': 1},  # Jump
            {'attack': 1},  # Chop tree
            {"camera": "Box(low=-180.0, high=180.0, shape=(2,))"}
        ]
        self.action_space = Discrete(len(self._actions))

    def action(self, action_idx):
        return self._actions[action_idx]

# Define Reward Wrapper with First-Time Inventory Entry Tracking
class MineRLRewardWrapper(RewardWrapper):
    def __init__(self, env, task):
        super().__init__(env)
        self.task = task
        self.collected_items = set()
        self.prev_position = None
    
    def reward(self, reward):
        obs = self.env.unwrapped.observation_space.sample()  # Retrieve the last observation
        inventory = obs.get('inventory', {})
        position = obs.get('location', [0, 0, 0])
        min_move_threshold = 1.2
        
        if self.prev_position is not None:
            distance_moved = np.linalg.norm(np.array(position) - np.array(self.prev_position))
            if distance_moved > min_move_threshold:
                reward += distance_moved * 0.1  # Reward based on meters moved
                print("reward given for distance moved")
                self.prev_position = position
        else:
            self.prev_position = position

        if self.task == "wood_collection":
            if 'oak_log' in inventory and 'oak_log' not in self.collected_items:
                reward += 1
                self.collected_items.add('oak_log')
            if 'birch_log' in inventory and 'birch_log' not in self.collected_items:
                reward += 1
                self.collected_items.add('birch_log')
            
            if 'planks' in inventory and 'planks' not in self.collected_items:
                reward += 2
                self.collected_items.add('planks')
            if 'stick' in inventory and 'stick' not in self.collected_items:
                reward += 4
                self.collected_items.add('stick')
        return reward

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        pov_space = observation_space.spaces["pov"]

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            sample_pov = th.zeros(1, *pov_space.shape)  # Simulate a batch with 1 image
            n_flatten = self.cnn(sample_pov).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))

def create_env(max_episode_steps=1000):
    env = gym.make("MineRLTreechop-v0")
    env.make_interactive(port=None, realtime=False)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    # env = ChoptreeObservationWrapper(env)
    env = ChoptreeActionWrapper(env)
    # env = MineRLRewardWrapper(env, task)
    return DummyVecEnv([lambda: env])


def train_agent(experiment_name, timesteps, max_episode_steps=1000, video_trigger_steps=500, video_length=100, model_path=None):
    experiment_logdir = f"./tensorboard_logs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(experiment_logdir, exist_ok=True)

    env = create_env(max_episode_steps)
    video_trigger = lambda step: step % video_trigger_steps == 0
    env = TensorboardVideoRecorder(env, video_trigger=video_trigger, video_length=video_length, tb_log_dir=experiment_logdir)

    policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=256))
    
    if model_path:
        model = PPO.load(model_path, env=env, device=device)
    else:
        model = PPO('MultiInputPolicy', env, verbose=1,
                    learning_rate=3e-3, 
                    policy_kwargs=policy_kwargs, tensorboard_log=experiment_logdir, device=device)
    
    model.learn(total_timesteps=timesteps)
    model.save(f"models/choptree_model.zip")
    return f"models/choptree_model.zip"

experiment_name = "Choptree_Training_Test_3"

max_episode_steps = 1000
video_trigger_steps = 1000
video_length = 200

model_path = train_agent(experiment_name, timesteps=5000, max_episode_steps=max_episode_steps,
                         video_trigger_steps=video_trigger_steps, video_length=video_length)
