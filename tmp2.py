import gym
from matplotlib import pyplot as plt
import numpy as np
env = gym.make("Pong-v0")

obs = env.reset()

plt.imshow(obs)
plt.show()