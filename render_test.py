import minerl
import gym
from PIL import Image
import numpy as np


env = gym.make('MineRLObtainDiamondShovel-v0')
env.make_interactive(port=None, realtime=False)
obs = env.reset()

frames = []

for _ in range(10):
    frame = env.render()
    frames.append(frame)

if not frames:
    print("No frames recorded!")
else:
    video_np = np.array(frames, dtype=np.uint8)  # Shape: (T, H, W, C)


print("Total frames captured:", len(frames))
print("Frame shape:", frames[0].shape if frames else "No frames captured")
print(video_np)
Image.fromarray(frames[0]).save('frame.png')

