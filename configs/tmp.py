import gym
from matplotlib import pyplot as plt
import numpy as np


env = gym.make("Pong-v0", frameskip=1)
ob = env.reset()
print(env.get_action_meanings())
for i in range(58):
	env.step(1)
# ob, _, _, _ = env.step(1)
l = []




# for
ob, _, _, _ = env.step(1)
ob = ob[34:160 + 34]
print(ob.shape)

# ob[np.where(np.all(ob == player, axis=-1))] = (0,0,0)
l.append(ob)

final = np.concatenate(l, axis=1)


plt.imshow(final)
plt.show()