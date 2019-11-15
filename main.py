import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import random
from PolicyGradient import REINFORCE
import os
from tensorflow import keras
from tensorflow.keras import layers

def nn_model(input_shape, output_shape, convolutional=False):
	model = keras.Sequential()
	if convolutional:
		model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
		#model.add(layers.MaxPooling2D((2, 2)))
		#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.Flatten())
	else:
		model.add(layers.Dense(10, input_shape=input_shape, activation='relu'))
	model.add(layers.Dense(20, activation='relu'))
	model.add(layers.Dense(output_shape))
	return model




if __name__ == '__main__':
	print(tf.__version__)
	tf.keras.backend.set_floatx('float64')
	os.system("rm -rf mylogs/*")
	# Generate environment
	env_name = 'MountainCar-v0'

	env = gym.make(env_name)


	if env_name == 'MountainCar-v0':
		policy_model = nn_model(env.observation_space.shape, env.action_space.n)
		value_model = nn_model(env.observation_space.shape, 1)
	elif env_name == 'Pong-v0':
		policy_model = nn_model(env.observation_space.shape, env.action_space.n, convolutional=True)
		value_model = nn_model(env.observation_space.shape, 1, convolutional=True)
	else:
		raise NotImplementedError(f"Not implemented environment {env_name}")

	# For visualization
	# You provide the directory to write to (can be an existing
	# directory, including one with existing data -- all monitor files
	# will be namespaced). You can also dump to a tempdir if you'd
	# like: tempfile.mkdtemp().
	outdir = '/tmp/random-agent-results'
	#env = wrappers.Monitor(env, directory=outdir, force=True)
	
	# Set Random Seed, probably not needed but useful for reproduction
	seed = 0
	env.seed(seed)
	tf.random.set_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	
	policy_model.summary()

	agent = REINFORCE(env, 1e-2, .95, policy_model)
	episodes = 10000
	agent.train(episodes)

	
	# Close the env and write monitor result info to disk
	env.close()