import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gym
from utils import nn_model
from matplotlib import pyplot as plt
import time
config = {
	"batch_size" : 10000,
	"epsilon" : 0.65,
	"n_paths" : 15,
	"correlated_epsilon":False
}

class ChangeActionSpaceEnv(gym.Wrapper):
	def __init__(self, env):
		"""Take action on reset for environments that are fixed until firing."""
		gym.Wrapper.__init__(self, env)
		self.action_space = gym.spaces.Discrete(3)
		self.observation_space = gym.spaces.Box(low = 0, high =160, dtype=np.uint8, shape=(3,))

	def get_ob(self, ob):
		ob = ob[34:160 + 34]
		ball = (236,236,236)
		player = (92, 186, 92)
		ys, xs = np.where(np.all(ob == ball, axis=-1))
		if len(xs) > 0:
			ball_x = xs[0]
			ball_y = ys[0]
		else:
			ball_x = 0
			ball_y = 0
		# ob[np.where(np.all(ob == ball, axis=-1))] = (255,0,0)
		ys, xs = np.where(np.all(ob == player, axis=-1))
		# player_x = xs[0]
		player_y = ys[0]
		new_ob = np.array([ball_x, ball_y, player_y])
		return new_ob

	def reset(self):
		self.env.reset()
		for i in range(29):
			self.env.step(1)
		ob, _, _, _ = self.env.step(1)
		return self.get_ob(ob)

	def step(self, ac):
		if ac == 0:
			ac = 2
		# else:
		# 	ac = 3
		elif ac ==2:
			ac = 3
		else:
			ac = 0
		ob, r, done, info = self.env.step(ac)
		ob = self.get_ob(ob)
		while not done and ob[0] == -1 and ob[1] == -1:
			ob, r, done, info = self.env.step(ac)
			ob = self.get_ob(ob)
		return (ob, r, done, info)

env = gym.make("Pong-v4", frameskip=2)
env = ChangeActionSpaceEnv(env)

policy_model = nn_model((3,), 3)
value_model = nn_model((3,), 1)


# def policy(ob):
# 	if ob[1] < ob[2]:
# 		return 0
# 	elif ob[1] > ob[2]:
# 		return 2
# 	else:
# 		return 1