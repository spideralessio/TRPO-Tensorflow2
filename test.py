import tensorflow as tf
from TRPO import TRPO
import argparse
import importlib

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
	
	mod = importlib.import_module(f"configs.{env_name}")
	
	print("Playing in", env_name)

	policy_model = mod.policy_model
	env = mod.env
	agent = TRPO(env_name, env, policy_model, epsilon=0, **mod.config)
	episodes = args.episodes
	agent.load_weights(args.ckpt)
	agent.render_episode(episodes)
	agent.close()