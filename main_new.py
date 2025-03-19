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
    def __init__(self, env):
        super().__init__(env)

        # Define the target action space
        self.TARGET_ACTION_SPACE = {
            "ESC": Discrete(2),
            "attack": Discrete(2),
            "back": Discrete(2),
            "camera": Box(low=-180.0, high=180.0, shape=(2,)),
            "drop": Discrete(2),
            "forward": Discrete(2),
            "hotbar.1": Discrete(2),
            "hotbar.2": Discrete(2),
            "hotbar.3": Discrete(2),
            "hotbar.4": Discrete(2),
            "hotbar.5": Discrete(2),
            "hotbar.6": Discrete(2),
            "hotbar.7": Discrete(2),
            "hotbar.8": Discrete(2),
            "hotbar.9": Discrete(2),
            "inventory": Discrete(2),
            "jump": Discrete(2),
            "left": Discrete(2),
            "pickItem": Discrete(2),
            "right": Discrete(2),
            "sneak": Discrete(2),
            "sprint": Discrete(2),
            "swapHands": Discrete(2),
            "use": Discrete(2)
        }

        # Count discrete and continuous actions
        self.discrete_keys = [k for k, v in self.TARGET_ACTION_SPACE.items() if isinstance(v, Discrete)]
        self.camera_keys = [k for k, v in self.TARGET_ACTION_SPACE.items() if isinstance(v, Box)]

        # Number of discrete and continuous actions
        discrete_actions = len(self.discrete_keys)
        camera_actions = sum(v.shape[0] for k, v in self.TARGET_ACTION_SPACE.items() if isinstance(v, Box))

        # Define new action space as Box
        self.action_space = Box(
            low=np.array([0] * discrete_actions + [-180.0] * camera_actions),
            high=np.array([1] * discrete_actions + [180.0] * camera_actions),
            dtype=np.float32
        )

    def action(self, action_array):
        """
        Convert the flattened action array back to dictionary format.
        """
        action_dict = {}
        index = 0

        # Process discrete actions
        for key in self.discrete_keys:
            action_dict[key] = int(action_array[index] > 0.5)  # Convert float [0,1] to binary {0,1}
            index += 1

        # Process camera actions
        for key in self.camera_keys:
            action_dict[key] = action_array[index:index + 2]  # Extract the 2D camera movement
            index += 2  # Move index forward

        return action_dict


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
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Extract inventory size
        inventory_size = observation_space.spaces["inventory"].shape[0]

        # Define CNN for processing images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        # Get CNN output size dynamically
        with th.no_grad():
            sample_input = th.zeros(1, 3, 360, 640)  # Simulate batch with 1 image
            n_flatten = self.cnn(sample_input).shape[1]

        # MLP for inventory
        self.inventory_fc = nn.Sequential(
            nn.Linear(inventory_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )

        # Final FC layer to merge CNN and inventory
        self.final_fc = nn.Sequential(
            nn.Linear(n_flatten + 32, features_dim), nn.ReLU()
        )

    def forward(self, observations):
        # Process the image (pov)
        image_features = self.cnn(observations["pov"])

        # Process the inventory vector
        inventory_features = self.inventory_fc(observations["inventory"].float())

        # Concatenate both processed features
        combined = th.cat([image_features, inventory_features], dim=1)

        return self.final_fc(combined)



def create_env(task, max_episode_steps=1000):
    env = gym.make("MineRLObtainDiamondShovel-v0")
    env.make_interactive(port=None, realtime=False)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = MineRLObservationWrapper(env)
    env = MineRLActionWrapper(env)
    env = MineRLRewardWrapper(env, task)

    print("\n--- DEBUG: ENVIRONMENT SPACES ---")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    print("---------------------------------\n")

    return DummyVecEnv([lambda: env])


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

steps = [
    # ("movement", 50000, 500, None),  # Learning movement
    ("wood_collection", 15000, 2000, None),  # Learning to collect wood
    # ("craft_tools", 30000, "models/wood_collection.zip"),  # Crafting planks, sticks, pickaxe
    # ("mine_stone", 40000, "models/craft_tools.zip"),  # Mining stone and crafting stone pickaxe
    # ("mine_iron", 50000, "models/mine_stone.zip"),  # Mining iron and crafting iron pickaxe
    # ("mine_diamonds", 60000, "models/mine_iron.zip"),  # Finding and mining diamonds
    # ("craft_diamond_pickaxe", 70000, "models/mine_diamonds.zip")  # Crafting diamond pickaxe
]

experiment_name = "1New_Wood_test_1"

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
