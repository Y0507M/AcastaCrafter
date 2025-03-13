import gym
import minerl
import numpy as np
import os

from datetime import datetime
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from tensorboard_video_recorder import TensorboardVideoRecorder


class ObservationShaping(gym.ObservationWrapper):
     def __init__(self, env):
         super().__init__(env)
         self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

     def observation(self, obs):
         return obs['pov']


class ActionShaping(gym.ActionWrapper):
    """
    The default MineRL action space is the following dict:

    Dict(attack:Discrete(2),
         back:Discrete(2),
         camera:Box(low=-180.0, high=180.0, shape=(2,)),
         craft:Enum(crafting_table,none,planks,stick,torch),
         equip:Enum(air,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         forward:Discrete(2),
         jump:Discrete(2),
         left:Discrete(2),
         nearbyCraft:Enum(furnace,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         nearbySmelt:Enum(coal,iron_ingot,none),
         place:Enum(cobblestone,crafting_table,dirt,furnace,none,stone,torch),
         right:Discrete(2),
         sneak:Discrete(2),
         sprint:Discrete(2))

    It can be viewed as:
         - buttons, like attack, back, forward, sprint that are either pressed or not.
         - mouse, i.e. the continuous camera action in degrees. The two values are pitch (up/down), where up is
           negative, down is positive, and yaw (left/right), where left is negative, right is positive.
         - craft/equip/place actions for items specified above.
    So an example action could be sprint + forward + jump + attack + turn camera, all in one action.

    This wrapper makes the action space much smaller by selecting a few common actions and making the camera actions
    discrete. You can change these actions by changing self._actions below. That should just work with the RL agent,
    but would require some further tinkering below with the BC one.
    """
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            # [('back', 1)],
            # [('left', 1)],
            # [('right', 1)],
            # [('jump', 1)],
            # [('forward', 1), ('attack', 1)],
            # [('craft', 'planks')],
            [('forward', 1), ('jump', 1)],
            [('camera', [-self.camera_angle, 0])],
            [('camera', [self.camera_angle, 0])],
            [('camera', [0, self.camera_angle])],
            [('camera', [0, -self.camera_angle])],
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


def train_agent(experiment_name, timesteps, max_episode_steps=1000, video_trigger_steps=500, video_length=100, model_path=None):
    experiment_logdir = f"./tensorboard_logs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(experiment_logdir, exist_ok=True)

    env = gym.make("MineRLTreechop-v0")
    env.make_interactive(port=None, realtime=False)
    env = gym.wrappers.time_limit.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = ObservationShaping(env)
    env = ActionShaping(env)

    video_trigger = lambda step: step % video_trigger_steps == 0
    env = TensorboardVideoRecorder(env, video_trigger=video_trigger, video_length=video_length, tb_log_dir=experiment_logdir)

    #policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=256))
    
    if model_path:
        model = PPO2.load(model_path, env=env)
    else:
        model = PPO2('CnnPolicy', env, verbose=1,
                    learning_rate=3e-3,
                    tensorboard_log=experiment_logdir)

    # pretrain
    dataset = ExpertDataset(expert_path='expert/MineRLTreechop-v0/expert.npz')
    model.pretrain(dataset)

    model.learn(total_timesteps=timesteps)
    model.save(f"model_{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")


experiment_name = "treechop"

max_episode_steps = 1000
video_trigger_steps = 1000
video_length = 200

if __name__ == '__main__':
    train_agent(experiment_name,
                timesteps=5000,
                max_episode_steps=max_episode_steps,
                video_trigger_steps=video_trigger_steps,
                video_length=video_length)


