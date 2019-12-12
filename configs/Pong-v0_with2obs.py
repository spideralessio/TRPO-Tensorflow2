import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gym
from utils import nn_model
from matplotlib import pyplot as plt
config = {
	"epsilon_decay" : lambda x: max(0.1, x - 1e-3),
	"n_paths" : 20,
	"gamma" : 0.99,
	"batch_size" :10000,
}

class ChangeActionSpaceEnv(gym.Wrapper):
	def __init__(self, env):
		"""Take action on reset for environments that are fixed until firing."""
		gym.Wrapper.__init__(self, env)
		self.action_space = gym.spaces.Discrete(3)
		self.observation_space = gym.spaces.Box(low = 0, high =160, dtype=np.uint8, shape=(6,))
		self.last_ob = None		

	def get_ob(self, ob):
		try:
			ob = ob[34:160 + 34]
			ball = (236,236,236)
			player = (92, 186, 92)
			xs, ys = np.where(np.all(ob == ball, axis=-1))
			if len(xs) > 0:
				ball_x = xs[0]
				ball_y = ys[0]
			else:
				ball_x = 0
				ball_y = 0
			# ob[np.where(np.all(ob == ball, axis=-1))] = (255,0,0)
			xs, ys = np.where(np.all(ob == player, axis=-1))
			# player_x = xs[0]
			player_y = ys[0]
			new_ob = np.array([ball_x, ball_y, player_y])
			last_ob = self.last_ob
			self.last_ob = new_ob
			if last_ob is not None:
				return np.concatenate((new_ob, last_ob))
			else:
				return np.concatenate((new_ob, new_ob))
		except:
			plt.imshow(ob)
			plt.show()

	def reset(self):
		self.last_ob = None
		self.env.reset()
		for i in range(58):
			self.env.step(1)
		ob, _, _, _ = self.env.step(1)
		return self.get_ob(ob)

	def step(self, ac):
		if ac == 0:
			ac = 2
		elif ac ==2:
			ac = 3
		else:
			ac = 0
		ob, r, done, info = self.env.step(ac)
		ob = self.get_ob(ob)
		return (ob, r, done, info)


# def nn_model(input_shape, output_shape):
# 	model = keras.Sequential()
# 	model.add(layers.Lambda(lambda x: tf.cast(tf.image.resize(tf.image.rgb_to_grayscale(tf.image.crop_to_bounding_box(x, 33,0,160,160)), size=(64,64)), dtype=tf.float64)/256., input_shape=input_shape))
# 	# model.add(layers.Lambda(lambda x: tf.image.rgb_to_grayscale(tf.cast(x, dtype=tf.float64)/255.), input_shape=input_shape))
# 	model.add(layers.Flatten())
# 	model.add(layers.Dense(128, input_shape=input_shape, activation='relu'))
# 	model.add(layers.Dense(64, activation='relu'))
# 	model.add(layers.Dense(output_shape))
# 	return model

env = gym.make("Pong-v0", difficulty=0, frameskip=1)
env = ChangeActionSpaceEnv(env)

policy_model = nn_model((6,), 3)
value_model = nn_model((6,), 1)
