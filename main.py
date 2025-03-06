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

# class MineRLObservationWrapper(ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         # Grayscale image: 64x64
#         self.observation_space = Box(low=0, high=255, shape=(360, 640, 3), dtype=np.uint8)

#     def observation(self, obs):
#         pov = obs['pov']
#         gray_pov = cv2.cvtColor(pov, cv2.COLOR_RGB2GRAY)
#         resized_pov = cv2.resize(gray_pov, (64, 64), interpolation=cv2.INTER_AREA)
#         resized_pov = np.expand_dims(resized_pov, axis=0)
#         return resized_pov

class MineRLObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Extract inventory keys dynamically
        sample_obs = env.observation_space.sample()
        self.inventory_keys = list(sample_obs['inventory'].keys())

        # Define new observation space
        inventory_size = len(self.inventory_keys)  # Number of inventory items
        pov_shape = (3, 360, 640)  # Keep full RGB image

        # Separate observation spaces for image and inventory
        self.observation_space = gym.spaces.Dict({
            "pov": Box(low=0, high=255, shape=pov_shape, dtype=np.uint8),  # Image
            "inventory": Box(low=0, high=2304, shape=(inventory_size,), dtype=np.uint16)  # Flattened inventory
        })

    def observation(self, obs):
        # Process inventory
        inventory_values = np.array([obs['inventory'][key] for key in self.inventory_keys], dtype=np.uint16)

        # Process POV image
        pov = obs['pov']
        pov = cv2.resize(pov, (640, 360), interpolation=cv2.INTER_LINEAR)  # Resize if necessary
        pov = np.transpose(pov, (2, 0, 1))  # Convert from (H, W, C) → (C, H, W)

        return {"pov": pov, "inventory": inventory_values}


class MineRLActionWrapper(ActionWrapper):
    def __init__(self, env, task):
        super().__init__(env)
        self.hold_attack_steps = 90
        self.attack_counter = 0

        if task == "movement":
            self._actions = [
                {'forward': 1}, {'jump': 1}, {'camera': [0, 5]}, {'camera': [0, -5]}
            ]  # Only movement actions
        elif task == "wood_collection":
            self._actions = [
                {'forward': 1}, {'jump': 1}, {'attack': 1}, {'break': 1}, {'camera': [0, 5]}, {'camera': [0, -5]}
            ]
        else:
            self._actions = [
                {'forward': 1},                               # Move forward
                {'attack': 1},                                # Mine/attack
                {'jump': 1},                                  # Jump
                {'craft': 'planks'},                          # Craft planks
                {'craft': 'stick'},                           # Craft sticks
                {'craft': 'crafting_table'},                  # Craft crafting table
                {'craft': 'wooden_pickaxe'},                  # Craft wooden pickaxe
                {'craft': 'stone_pickaxe'},                   # Craft stone pickaxe
                {'craft': 'furnace'},                         # Craft furnace
                {'craft': 'iron_pickaxe'},                    # Craft iron pickaxe
                {'craft': 'diamond_shovel'},                  # Craft diamond shovel
                {'equip': 'wooden_pickaxe'},                  # Equip wooden pickaxe
                {'equip': 'stone_pickaxe'},                   # Equip stone pickaxe
                {'equip': 'iron_pickaxe'},                    # Equip iron pickaxe
                {'camera': [0, 5]},                           # Look right
                {'camera': [0, -5]},                          # Look left
            ]
        self.action_space = Discrete(len(self._actions))

    def action(self, action_idx):
        if self.attack_counter > 0:
            # If in attack-hold mode, keep returning attack action
            self.attack_counter -= 1
            return {'attack': 1}

        selected_action = self._actions[action_idx]

        if 'break' in selected_action:
            # Start holding attack for `hold_attack_steps` steps (default 90)
            self.attack_counter = self.hold_attack_steps - 1
            return {'attack': 1}

        return selected_action

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
        elif self.task == "craft_tools":
            if 'crafting_table' in inventory and 'crafting_table' not in self.collected_items:
                reward += 4
                self.collected_items.add('crafting_table')
            if 'wooden_pickaxe' in inventory and 'wooden_pickaxe' not in self.collected_items:
                reward += 8
                self.collected_items.add('wooden_pickaxe')
        elif self.task == "mine_stone":
            if 'cobblestone' in inventory and 'cobblestone' not in self.collected_items:
                reward += 16
                self.collected_items.add('cobblestone')
            if 'furnace' in inventory and 'furnace' not in self.collected_items:
                reward += 32
                self.collected_items.add('furnace')
            if 'stone_pickaxe' in inventory and 'stone_pickaxe' not in self.collected_items:
                reward += 32
                self.collected_items.add('stone_pickaxe')
        elif self.task == "mine_iron":
            if 'iron_ore' in inventory and 'iron_ore' not in self.collected_items:
                reward += 64
                self.collected_items.add('iron_ore')
            if 'iron_ingot' in inventory and 'iron_ingot' not in self.collected_items:
                reward += 64
                self.collected_items.add('iron_ingot')
            if 'iron_pickaxe' in inventory and 'iron_pickaxe' not in self.collected_items:
                reward += 256
                self.collected_items.add('iron_pickaxe')
        elif self.task == "mine_diamonds":
            if 'diamond' in inventory and 'diamond' not in self.collected_items:
                reward += 1024
                self.collected_items.add('diamond')
        elif self.task == "craft_diamond_shovel":
            if 'diamond_shovel' in inventory and 'diamond_shovel' not in self.collected_items:
                reward += 2048
                self.collected_items.add('diamond_shovel')
        return reward

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Get inventory feature size
        inventory_size = observation_space.spaces["inventory"].shape[0]

        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            sample_input = th.zeros(1, 3, 360, 640)  # Simulate batch with 1 image
            n_flatten = self.cnn(sample_input).shape[1]

        # Fully connected layers for inventory
        self.inventory_fc = nn.Sequential(
            nn.Linear(inventory_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )

        # Combine CNN and inventory outputs
        self.final_fc = nn.Sequential(
            nn.Linear(n_flatten + 32, features_dim), nn.ReLU()
        )

    def forward(self, observations):
        # Process image
        image_features = self.cnn(observations["pov"])

        # Process inventory
        inventory_features = self.inventory_fc(observations["inventory"].float())

        # Concatenate both
        combined = th.cat([image_features, inventory_features], dim=1)

        return self.final_fc(combined)

def create_env(task, max_episode_steps=1000):
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env.make_interactive(port=None, realtime=False)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = MineRLObservationWrapper(env)
    env = MineRLActionWrapper(env, task)
    env = MineRLRewardWrapper(env, task)

    print("\n--- DEBUG: ENVIRONMENT SPACES ---")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    print("---------------------------------\n")

    return DummyVecEnv([lambda: env])

# env = gym.make("MineRLObtainDiamondShovel-v0")
# # Test: to allow recording
# env.make_interactive(port=None, realtime=False)

# # Limit to ~50-60 seconds
# env = TimeLimit(env, max_episode_steps=1000)
# env = MineRLObservationWrapper(env)
# env = MineRLActionWrapper(env)


def train_agent(experiment_name, task_name, timesteps, max_episode_steps=1000, video_trigger_steps=500, video_length=100, model_path=None):
    experiment_logdir = f"./minerl_tensorboard/{experiment_name}_gaog5_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/{task_name}"
    os.makedirs(experiment_logdir, exist_ok=True)

    env = create_env(task_name, max_episode_steps)
    video_trigger = lambda step: step % video_trigger_steps == 0
    env = TensorboardVideoRecorder(
        env=env,
        video_trigger=video_trigger,
        video_length=video_length,  # number of frames to record
        tb_log_dir=experiment_logdir
    )
    policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=256))
    
    if model_path:
        model = PPO.load(model_path, env=env, device=device)
    else:
        model = PPO('MultiInputPolicy', env, verbose=1,
                    learning_rate=3e-3, 
                    policy_kwargs=policy_kwargs, tensorboard_log=experiment_logdir, device=device)
    
    model.learn(total_timesteps=timesteps)
    model.save(f"models/{task_name}.zip")
    return f"models/{task_name}.zip"

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=256),
# )

# env = DummyVecEnv([lambda: env])

# video_trigger = lambda step: step % 500 == 0
# env = TensorboardVideoRecorder(
#     env=env,
#     video_trigger=video_trigger,
#     video_length=100,  # number of frames to record
#     tb_log_dir=experiment_logdir
# )

# model = PPO(
#     'CnnPolicy',
#     env,
#     verbose=1,
#     policy_kwargs=policy_kwargs,
#     tensorboard_log=experiment_logdir,
#     device=device
# )

# # Train for an initial short run to test
# print("Model Learning")
# model.learn(total_timesteps=2000)

# # Save the model
# model.save("minerl_diamond_shovel_ppo")

steps = [
    # ("movement", 50000, 500, None),  # Learning movement
    ("wood_collection", 15000, 2000, None),  # Learning to collect wood
    # ("craft_tools", 30000, "models/wood_collection.zip"),  # Crafting planks, sticks, pickaxe
    # ("mine_stone", 40000, "models/craft_tools.zip"),  # Mining stone and crafting stone pickaxe
    # ("mine_iron", 50000, "models/mine_stone.zip"),  # Mining iron and crafting iron pickaxe
    # ("mine_diamonds", 60000, "models/mine_iron.zip"),  # Finding and mining diamonds
    # ("craft_diamond_pickaxe", 70000, "models/mine_diamonds.zip")  # Crafting diamond pickaxe
]

experiment_name = "Wood_test_3"

max_episode_steps = 1000
video_trigger_steps = 1000
video_length = 300

# Train Agent Sequentially
for task_name, timesteps, max_episode_steps, model_path in steps:
    print(f"Training task: {task_name}")
    model_path = train_agent(experiment_name, task_name, timesteps, 
                                max_episode_steps=max_episode_steps, 
                                video_trigger_steps = video_trigger_steps, 
                                video_length = video_length, model_path=model_path)

    # model = PPO.load(model_path, env=env, device=device)
    # obs = env.reset()
    # total_reward = 0
    # for _ in range(max_episode_steps):
    #     action, _ = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     total_reward += reward
    #     if done:
    #         break
    # print(f"Evaluation Reward for {task_name}: {total_reward}")

# print("Training completed!")

# obs = env.reset()
# done = False
# total_reward = 0

# print("Model Predicting")
# while not done:
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     total_reward += reward
#     # env.render()

# print("Total reward:", total_reward)
