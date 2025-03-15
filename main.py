import gym
import minerl
import numpy as np
import os

from datetime import datetime
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from tensorboard_video_recorder import TensorboardVideoRecorder

from wrappers import ObservationShaping, TreechopActionWrapper


def train_agent(experiment_name, video_trigger_steps=500, video_length=100):
    experiment_logdir = f"./tensorboard_logs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(experiment_logdir, exist_ok=True)

    env = gym.make("MineRLTreechop-v0")
    env.make_interactive(port=None, realtime=False)
    env = ObservationShaping(env)
    env = TreechopActionWrapper(env)

    video_trigger = lambda step: step % video_trigger_steps == 0
    env = TensorboardVideoRecorder(env, video_trigger=video_trigger, video_length=video_length, tb_log_dir=experiment_logdir)

    model = PPO2('CnnPolicy', env, verbose=1,
                 learning_rate=0.001,
                 tensorboard_log=experiment_logdir)

    # pretrain
    #dataset = ExpertDataset(expert_path='expert/MineRLTreechop-v0/expert.npz',
    #                        batch_size=128,
    #                        randomize=False)
    #model.pretrain(dataset)

    model.learn(total_timesteps=1_000_000)
    model.save(f"model_{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip")


experiment_name = "treechop"

video_trigger_steps = 2e4
video_length = 2000

if __name__ == '__main__':
    train_agent(experiment_name,
                video_trigger_steps=video_trigger_steps,
                video_length=video_length)

