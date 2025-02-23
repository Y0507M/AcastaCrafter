import time
import os

from craftium import CraftiumEnv

env = CraftiumEnv(
    env_dir="craftium-envs/small-room",
    render_mode="human",
    obs_width=512,
    obs_height=512,
    minetest_dir=os.getcwd(),
)

iters = 1000

observation, info = env.reset()

ep_ret = 0
start = time.time()
for i in range(iters):
    action = dict() # empty action (do nothing)
    observation, reward, terminated, truncated, _info = env.step(action)

    ep_ret += reward
    print(i, reward, terminated, truncated, ep_ret)

    if terminated or truncated:
        observation, info = env.reset()
        ep_ret = 0

end = time.time()
print(f"** {iters} frames in {end-start}s => {(end-start)/iters} per frame")

env.close()