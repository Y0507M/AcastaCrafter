import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
import minerl
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from tensorboard_video_recorder import TensorboardVideoRecorder  # Import the custom video recorder

from datetime import datetime

# ✅ Step 1: Load the Behavior Cloning (BC) Model
from treechop_bc import BCModel


class MineRLActionWrapper(gym.ActionWrapper):
    """
    Converts MineRL's Dict action space into a single MultiDiscrete space for PPO.
    - Converts 8 binary actions + 2 continuous camera actions into a flattened MultiDiscrete space.
    """
    def __init__(self, env, camera_bins=11):
        super(MineRLActionWrapper, self).__init__(env)

        self.camera_bins = camera_bins  # Number of bins for discretizing camera movement
        self.camera_min = -180.0
        self.camera_max = 180.0

        # Define new action space: [Discrete Actions] + [Discretized Camera]
        self.action_space = spaces.MultiDiscrete(
            [2] * 8 + [self.camera_bins] * 2  # 8 binary actions + 2 camera bins
        )

    def action(self, action):
        """
        Convert PPO's MultiDiscrete output back into MineRL's Dict action format.
        """
        minerl_action = {
            "attack": int(action[0]),
            "back": int(action[1]),
            "forward": int(action[2]),
            "jump": int(action[3]),
            "left": int(action[4]),
            "right": int(action[5]),
            "sneak": int(action[6]),
            "sprint": int(action[7]),
            "camera": np.array([
                np.interp(action[8], [0, self.camera_bins - 1], [self.camera_min, self.camera_max]),
                np.interp(action[9], [0, self.camera_bins - 1], [self.camera_min, self.camera_max])
            ], dtype=np.float32)
        }
        return minerl_action

class MineRLObservationWrapper(gym.ObservationWrapper):
    """
    Extracts only the `pov` key from the observation dictionary and normalizes it.
    """
    def __init__(self, env):
        super(MineRLObservationWrapper, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def observation(self, obs):
        return obs["pov"]  # Return only the RGB image

def load_bc_model(model_path, device):
    bc_model = BCModel().to(device)
    bc_model.load_state_dict(torch.load(model_path, map_location=device))
    bc_model.eval()
    return bc_model

# ✅ Step 2: Define a Custom Feature Extractor for PPO
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, pretrained_model, features_dim=256):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()  # Set BC model to evaluation mode
        self.fc = nn.Linear(256, features_dim)  # Output layer for PPO

    def forward(self, x):
        with torch.no_grad():
            x = self.pretrained_model.conv(x)  # Use BC model’s convolutional layers
            x = x.view(x.size(0), -1)
            x = self.pretrained_model.fc(x)  # Extract learned features
        return self.fc(x)  # Pass through a final layer for PPO

# ✅ Step 3: Wrap the MineRL Environment in a Vectorized Wrapper for PPO
def make_env():
    env = gym.make("MineRLTreechop-v0")
    env = MineRLActionWrapper(env)
    env = MineRLObservationWrapper(env)
    return env

env = DummyVecEnv([make_env])  # Wrap in VecEnv

# ✅ Step 4: Wrap Environment with Tensorboard Video Recorder
log_dir = f"./tensorboard_bc_logs/treechop_bc_gaog5_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
env = TensorboardVideoRecorder(
    env=env,
    video_trigger=lambda step: step % 5000 == 0,  # Record every 5000 steps
    video_length=500,
    tb_log_dir=log_dir
)

# ✅ Step 5: Create Custom Callback for Logging to TensorBoard
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record("reward/mean_episode_reward", np.mean(self.locals["rewards"]))  # Log reward
        self.logger.record("training/loss", self.locals["loss"].item() if "loss" in self.locals else 0)
        return True

# ✅ Step 6: Load Pretrained BC Model and Set Up PPO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/treechop_bc_test_1/bc_model_epoch_10.pth"
bc_model = load_bc_model(model_path, device)

# Use BC model as a feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomFeatureExtractor,
    features_extractor_kwargs=dict(pretrained_model=bc_model, features_dim=256)
)

# ✅ Step 7: Initialize PPO Agent with TensorBoard Logging
new_logger = configure(log_dir, ["stdout", "tensorboard"])  # Log to TensorBoard

ppo = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)
ppo.set_logger(new_logger)

# ✅ Step 8: Train PPO with Logging
ppo.learn(total_timesteps=100000, callback=TensorboardCallback())

# ✅ Step 9: Save the Trained PPO Model
ppo.save("models/ppo_minerl")
print("✅ PPO training complete! Model saved.")
