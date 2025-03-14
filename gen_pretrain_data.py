import cv2
import gym
import minerl
import numpy as np
import os

ENV_ID = 'MineRLTreechop-v0'

NUM_INTERACTIONS = -1


image_folder = f'expert/{ENV_ID}/imgs'
os.makedirs(image_folder, exist_ok=True)


def convert_action(action, camera_margin=5):
    """
    attack          0
    forward         1
    forward+jump    2
    camera_1        3
    camera_2        4
    camera_3        5
    camera_4        6
    """
    if action['camera'][0] < -camera_margin:
        return 3
    elif action['camera'][0] > camera_margin:
        return 4
    elif action['camera'][1] < -camera_margin:
        return 5
    elif action['camera'][1] > camera_margin:
        return 6
    elif action['forward'] == 1:
        if action['jump'] == 1:
            return 2
        else:
            return 1
    elif action['attack'] == 1:
        return 0
    else:
        # No reasonable mapping (would be no-op)
        return -1


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
                return {'actions': np.array(expert_actions).reshape((len(expert_actions), 1)),
                        'obs': np.array(expert_obs, dtype=np.dtype('<U64')),
                        'rewards': np.array(expert_rewards),
                        'episode_returns': np.array(expert_episode_returns),
                        'episode_starts': np.array(expert_episode_starts)}

            act_ = convert_action(act)
            if act_ == -1:
                continue

            expert_actions.append(act_)

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

    return {'actions': np.array(expert_actions).reshape((len(expert_actions), 1)),
            'obs': np.array(expert_obs, dtype=np.dtype('<U64')),
            'rewards': np.array(expert_rewards),
            'episode_returns': np.array(expert_episode_returns),
            'episode_starts': np.array(expert_episode_starts)}

if __name__ == '__main__':
    expert = gen_expert()

    np.savez(f'expert/{ENV_ID}/expert.npz', **expert)

