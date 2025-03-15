import cv2
import gym
import minerl
import numpy as np
import os

ENV_ID = 'MineRLTreechop-v0'

NUM_INTERACTIONS = -1


image_folder = f'expert/{ENV_ID}/imgs'
os.makedirs(image_folder, exist_ok=True)

env = gym.make(ENV_ID)


def gen_expert():
    expert_actions = []
    expert_obs = []
    expert_rewards = []
    expert_episode_returns = []
    expert_episode_starts = []

    data = minerl.data.make(ENV_ID)
    trajectory_names = sorted(data.get_trajectory_names())
    count = 0
    for trajectory_name in trajectory_names:
        new_traj = True
        for obs, act, rew, _, done in data.load_data(trajectory_name):
            if NUM_INTERACTIONS != -1 and count >= NUM_INTERACTIONS:
                return {'actions': np.array(expert_actions),
                        'obs': np.array(expert_obs, dtype=np.dtype('<U64')),
                        'rewards': np.array(expert_rewards),
                        'episode_returns': np.array(expert_episode_returns),
                        'episode_starts': np.array(expert_episode_starts)}

            expert_actions.append(gym.spaces.utils.flatten(env.action_space, act))

            if new_traj:
                expert_episode_returns.append(0.0)
                expert_episode_starts.append(True)

                new_traj = False
            else:
                expert_episode_starts.append(False)

            obs_ = obs['pov'].squeeze().astype(np.float32)
            image_path = os.path.join(image_folder, "{}.{}".format(count + 1, 'jpg'))
            if obs_.shape[-1] == 3:
                obs_ = cv2.cvtColor(obs_, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, obs_)
            expert_obs.append(image_path)

            expert_rewards.append(rew)

            count += 1

    return {'actions': np.array(expert_actions),
            'obs': np.array(expert_obs, dtype=np.dtype('<U64')),
            'rewards': np.array(expert_rewards),
            'episode_returns': np.array(expert_episode_returns),
            'episode_starts': np.array(expert_episode_starts)}

if __name__ == '__main__':
    expert = gen_expert()

    np.savez(f'expert/{ENV_ID}/expert.npz', **expert)

