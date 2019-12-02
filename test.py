import gym
import tensorflow as tf
from TRPO import TRPO
from utils import nn_model
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Test TRPO Agent.")
	parser.add_argument('ckpt', help="Path to checkpoint. Ex: saved_models/TRPO-MountainCar-v0-Dec01_21-49-45/300.ckpt ")
	parser.add_argument('env', help="Environment used for training. ")
	parser.add_argument('--episodes', help="Number of episodes to test. (default 1)", default=1, type=int)
	args = parser.parse_args()


	print("Using Tensorflow", tf.__version__)
	tf.keras.backend.set_floatx('float64')
	# Generate environment
	env_name = args.env
	print("Playing in", env_name)

	env = gym.make(env_name)

	if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2']:
		policy_model = nn_model(env.observation_space.shape, env.action_space.n)
	elif env_name == 'Pong-v0':
		policy_model = nn_model(env.observation_space.shape, env.action_space.n, convolutional=True)
	else:
		raise NotImplementedError(f"Not implemented environment {env_name}")
	
	env.close()

	agent = TRPO(env_name, policy_model)
	episodes = args.episodes
	agent.load_weights(args.ckpt)
	agent.render_episode(episodes)
	agent.close()