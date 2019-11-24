import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import random
from PolicyGradient import REINFORCE
from TRPO2 import TRPO
import os
from utils import nn_model




if __name__ == '__main__':
	print(tf.__version__)
	tf.keras.backend.set_floatx('float64')
	# Generate environment
	env_name = 'LunarLander-v2'

	env = gym.make(env_name)
	if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2']:
		# if env_name == "MountainCar-v0":
			# env._max_episode_steps = 600
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
	# seed = 0
	# env.seed(seed)
	# tf.random.set_seed(seed)
	# np.random.seed(seed)
	# random.seed(seed)
	
	policy_model.summary()

	agent = TRPO(env, policy_model, value_model, render=False)
	episodes = 10000
	agent.train(episodes)

	
	# Close the env and write monitor result info to disk
	env.close()