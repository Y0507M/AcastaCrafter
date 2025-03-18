import gym
import minerl
import numpy as np
import torch
import torch.nn as nn
import os
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from gym import spaces

# debugging
import os
import time
import minerl
import logging
import socket

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure environment variables
os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = './minecraft_logs'
os.environ['MALMO_XMS'] = '2G'
os.environ['MALMO_XMX'] = '4G'
os.environ['MALMO_MINECRAFT_PORT'] = '10000'
os.environ['MINERL_HEADLESS'] = '1'
# try:
#     os.environ['MALMO_XSD_PATH'] = os.path.join(os.path.dirname(minerl.__file__), 'Malmo', 'Schemas')
# except Exception as e:
#     logger.warning(f"Could not set MALMO_XSD_PATH: {e}")

# Make sure the minecraft logs directory exists
os.makedirs('./minecraft_logs', exist_ok=True)

# Add a retry mechanism when creating the environment
def create_env_with_retry(env_id, max_retries=5):
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt+1}/{max_retries} to create environment")
            env = gym.make(env_id)
            # Add a longer wait after creation
            time.sleep(5)  # Let the environment stabilize
            return env
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            if attempt < max_retries - 1:
                logger.info("Waiting before retry...")
                time.sleep(20)  # Wait before retrying
            else:
                raise

# Create a wrapper to flatten the nested observation space
class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # New flattened observation space
        self.observation_space = spaces.Dict({
            "pov": env.observation_space.spaces["pov"],
            "compass_angle": spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
            "dirt_inventory": spaces.Box(low=0, high=2304, shape=(1,), dtype=np.int32)
        })
    
    def observation(self, observation):
        # Extract relevant information and flatten the structure
        try:
            return {
                "pov": observation["pov"],
                "compass_angle": np.array([observation["compass"]["angle"]], dtype=np.float32),
                "dirt_inventory": np.array([observation["inventory"]["dirt"]], dtype=np.int32)
            }
        except Exception as e:
            logger.error(f"Error processing observation: {e}")
            logger.info(f"Observation content: {observation}")
            raise
            

# Add an action wrapper to convert Box actions to Dict actions for MineRL
class MineRLActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Define a simplified action space that PPO can work with
        # Use Box space with 6 continuous values:
        # [forward_back, left_right, jump, attack, camera_x, camera_y]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0, 0, -1, -1]),
            high=np.array([1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )
    
    def action(self, action):
        # Convert the Box action to MineRL Dict action
        minerl_action = {
            "forward": 0,
            "back": 0,
            "left": 0,
            "right": 0,
            "jump": 0,
            "attack": 0,
            "camera": np.zeros(2, dtype=np.float32),
            "place": 0,
            "sneak": 0,
            "sprint": 0
        }
        
        # Forward/backward: action[0] in [-1,1] range
        if action[0] > 0.3:
            minerl_action["forward"] = 1
        elif action[0] < -0.3:
            minerl_action["back"] = 1
            
        # Left/right: action[1] in [-1,1] range
        if action[1] > 0.3:
            minerl_action["right"] = 1
        elif action[1] < -0.3:
            minerl_action["left"] = 1
        
        # Jump: action[2] in [0,1] range
        if action[2] > 0.5:
            minerl_action["jump"] = 1
            
        # Attack: action[3] in [0,1] range
        if action[3] > 0.5:
            minerl_action["attack"] = 1
            
        # Camera: action[4:6] in [-1,1] range, scale to appropriate camera movement
        # MineRL camera uses degrees, so we scale our normalized actions
        camera_x = action[4] * 10  # Scale to +/- 10 degrees horizontally
        camera_y = action[5] * 5   # Scale to +/- 5 degrees vertically
        minerl_action["camera"] = np.array([camera_x, camera_y], dtype=np.float32)
        
        return minerl_action

# Custom feature extractor for the flattened MineRL observations
class MineCraftFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Extract shapes
        self.pov_shape = observation_space.spaces["pov"].shape
        
        # Modified CNN for processing POV (image) with smaller kernels and strides
        self.cnn = nn.Sequential(
            # Use smaller kernels and strides to work with smaller images
            nn.Conv2d(self.pov_shape[2], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the output shape of the CNN
        with torch.no_grad():
            # Print the POV shape for debugging
            print(f"POV shape from observation space: {self.pov_shape}")
            sample_pov = torch.zeros(1, *self.pov_shape).permute(0, 3, 1, 2)  # Add batch dimension and reorder to NCHW
            print(f"Sample POV tensor shape: {sample_pov.shape}")
            cnn_output = self.cnn(sample_pov)
            cnn_output_shape = cnn_output.shape[1]
            print(f"CNN output shape: {cnn_output_shape}")
        
        # MLP for processing compass and inventory data, and combining with CNN output
        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_shape + 1 + 1, 256),  # +1 for compass angle, +1 for dirt inventory
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Process POV (image) through CNN
        # pov = observations["pov"].float() / 255.0  # Normalize to [0, 1]
        pov = torch.tensor(observations["pov"], dtype=torch.float32).to(self.device) / 255.0  # Normalize to [0, 1]
        pov = pov.permute(0, 3, 1, 2)  # NHWC -> NCHW format
        pov_features = self.cnn(pov)
        print(f"Processed POV tensor shape: {pov.shape}")
        
        # Extract compass angle and normalize to [-1, 1]
        compass_angle = observations["compass_angle"].float() / 180.0
        
        # Extract inventory and normalize
        dirt_count = observations["dirt_inventory"].float() / 2304.0
        
        # Combine all features
        combined = torch.cat([pov_features, compass_angle, dirt_count], dim=1)
        
        return self.mlp(combined)

def main():
    
    # And then use a custom port
    port = find_free_port()
    # os.environ['MALMO_MINECRAFT_PORT'] = '10000'
    
    # Create log directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # debugging
    env = create_env_with_retry("MineRLNavigate-v0")
    
    # Create a single environment with our flattened wrapper and action wrapper
    raw_env = gym.make("MineRLNavigate-v0")
    
    # Print observation space details for debugging
    print(f"Original observation space: {raw_env.observation_space}")
    print(f"POV shape: {raw_env.observation_space.spaces['pov'].shape}")
    
    env = FlattenObservationWrapper(raw_env)
    env = MineRLActionWrapper(env)  # Add action wrapper
    
    # Wrap it with DummyVecEnv for SB3 compatibility
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)  # Ensure images are in the right format
    
    # Set up callbacks for saving models during training
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="navigate_dense_model"
    )
    
    # Create policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=MineCraftFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        # Updated to use dict directly as recommended in the warning
        # net_arch=dict(pi=[256, 128], vf=[256, 128])
        net_arch = [256, 128] # Simplified to match the new feature extractor output
    )
    
    # Initialize the PPO agent
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Train the agent
    total_timesteps = 1000000  # Adjust based on your computational resources
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save the final model
    model.save(f"{models_dir}/navigate_dense_final")
    
    print("Training completed! Final model saved.")
    
    # Evaluate the trained agent
    mean_reward = evaluate_agent(model)
    print(f"Mean reward over evaluation episodes: {mean_reward}")

def evaluate_agent(model, num_episodes=10):
    """Evaluate the trained agent over a number of episodes."""
    raw_env = gym.make("MineRLNavigate-v0")
    env = FlattenObservationWrapper(raw_env)
    env = MineRLActionWrapper(env)  # Add action wrapper here too
    
    total_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        print(f"Sample observation: {obs}")
        done = False
        episode_reward = 0
        steps = 0
        max_steps = 8000  # Prevent too long evaluations
        
        while not done and steps < max_steps:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Optional: render the environment to watch the agent
            # env.render()
        
        # Check if the agent reached the goal
        if episode_reward >= 64:  # The environment gives +64 on success
            success_count += 1
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward}, Steps: {steps}")
    
    env.close()
    success_rate = (success_count / num_episodes) * 100
    print(f"Success rate: {success_rate}%")
    return sum(total_rewards) / len(total_rewards)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def test_env():
    """Test with a simpler environment."""
    try:
        # Try a different environment first
        print("Testing with a simpler environment...")
        simpler_env = gym.make("MineRLNavigate-v0")  # Try the simpler version
        print("Simple environment created successfully")
        
        obs = simpler_env.reset()
        print("Simple environment reset successfully")
        
        simpler_env.close()
        print("Simple environment closed successfully")
        
        # Then try the original environment
        # ...rest of your test code...
    except Exception as e:
        print(f"Error testing environment: {e}")
        import traceback
        traceback.print_exc()
        return False

# For running directly from command line
if __name__ == "__main__":
    print("Training agent for MineRLNavigate-v0")
    if torch.cuda.is_available():
        print("Using cuda device")
        
    # # Call this before main()
    # if not test_env():
    #     print("Environment test failed. Exiting.")
    #     exit(1)
    
    main()