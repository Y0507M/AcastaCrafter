import gym
import numpy as np


class TreechopActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        action[:4:2] = action[:4:2] > 0.5
        action[1:4:2] = 1 - action[:4:2]
        action[6::2] = action[6::2] > 0.5
        action[7::2] = 1 - action[6::2]
        return gym.spaces.utils.unflatten(self.env.action_space, action)


class ObservationShaping(gym.ObservationWrapper):
     def __init__(self, env):
         super().__init__(env)
         self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

     def observation(self, obs):
         return obs['pov']

