import gym
import tensorflow as tf
from TRPO import TRPO
from utils import nn_model
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Train TRPO Agent.")
	parser.add_argument('env', help="Environment used for training. ")
	parser.add_argument('--episodes', help="Number of episodes to train. (default 500)", default=500, type=int)
	parser.add_argument('--render', help="Render an episode at each training step", default=False, action="store_true")
	args = parser.parse_args()


	print("Using Tensorflow", tf.__version__)
	tf.keras.backend.set_floatx('float64')
	env_name = args.env
	print("Playing in", env_name)

	env = gym.make(env_name)

	if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2']:
		policy_model = nn_model(env.observation_space.shape, env.action_space.n)
		value_model = nn_model(env.observation_space.shape, 1)
	elif env_name == 'Pong-v0':
		policy_model = nn_model(env.observation_space.shape, env.action_space.n, convolutional=True)
		value_model = nn_model(env.observation_space.shape, 1, convolutional=True)
	else:
		raise NotImplementedError(f"Not implemented environment {env_name}")

	env.close()

	agent = TRPO(env_name, policy_model, value_model, render=args.render)
	episodes = args.episodes
	agent.train(episodes)
	agent.close()