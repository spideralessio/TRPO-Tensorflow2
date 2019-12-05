import tensorflow as tf
from TRPO import TRPO
import argparse
import importlib

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Train TRPO Agent.")
	parser.add_argument('env', help="Environment used for training. ")
	parser.add_argument('--episodes', help="Number of episodes to train. (default 500)", default=500, type=int)
	parser.add_argument('--render', help="Render an episode at each training step", default=False, action="store_true")
	args = parser.parse_args()


	print("Using Tensorflow", tf.__version__)
	tf.keras.backend.set_floatx('float64')
	env_name = args.env
	# https://github.com/openai/gym/wiki/Table-of-environments
	mod = importlib.import_module(f"configs.{env_name}")
	
	print("Playing in", env_name)

	policy_model = mod.policy_model
	value_model = mod.value_model

	agent = TRPO(env_name, policy_model, value_model, render=args.render, **mod.config)
	episodes = args.episodes
	agent.train(episodes)
	agent.close()