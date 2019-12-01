import tensorflow as tf
import gym
from utils import nn_model
import numpy as np
tf.keras.backend.set_floatx('float64')
env = gym.make("MountainCar-v0")
policy_model = nn_model(env.observation_space.shape, env.action_space.n)
policy_model.load_weights("mylogs/TRPO-MountainCar-v0-Dec01_21-49-45/340.ckpt")


def run_episode():
	ob = env.reset()
	done = False
	while not done:
		env.render()
		logits = policy_model(ob[np.newaxis, :])
		action = np.argmax(tf.nn.softmax(logits).numpy().ravel())
		ob, r, done, info = env.step(action)

while True:
	run_episode()