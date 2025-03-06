import gym
import minerl
import numpy as np
import cv2
from datetime import datetime

from gym import ObservationWrapper, ActionWrapper
from gym.spaces import Box, Discrete

import torch as th
import torch.nn as nn

device = 'cuda' if th.cuda.is_available() else 'cpu'

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from tensorboard_video_recorder import TensorboardVideoRecorder

UCNETID = "minh8"


class MineRLObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Grayscale image: 64x64
        self.observation_space = Box(low=0, high=255, shape=(1, 64, 64), dtype=np.uint8)

    def observation(self, obs):
        pov = obs['pov']
        gray_pov = cv2.cvtColor(pov, cv2.COLOR_RGB2GRAY)
        resized_pov = cv2.resize(gray_pov, (64, 64), interpolation=cv2.INTER_AREA)
        resized_pov = np.expand_dims(resized_pov, axis=0)
        return resized_pov

class MineRLActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
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
        return self._actions[action_idx]

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


def main():
    env = gym.make("MineRLTreechop-v0")
    # Test: to allow recording
    env.make_interactive(port=None, realtime=False)

    env = MineRLObservationWrapper(env)
    env = MineRLActionWrapper(env)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    env = DummyVecEnv([lambda: env])

    experiment_name = "ppo_minerl_treechop"
    experiment_logdir = f"./minerl_tensorboard/{experiment_name}_{UCNETID}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    video_trigger = lambda step: step % 100 == 0
    env = TensorboardVideoRecorder(
        env=env,
        video_trigger=video_trigger,
        video_length=100,  # number of frames to record
        tb_log_dir=experiment_logdir
    )

    model = PPO(
        'CnnPolicy',
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=experiment_logdir,
        device=device
    )

    # Train for an initial short run to test
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("ppo_minerl_treechop")

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    print("Total reward:", total_reward)


if __name__ == '__main__':
    main()

