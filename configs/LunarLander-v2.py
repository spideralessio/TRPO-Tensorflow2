from utils import nn_model2
import gym
import numpy as np
config = {
	"epsilon_decay" : lambda x: max(0.1, x - 5e-4),
	"n_paths" : 20
}

# if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2', 'Pong-ram-v0']:
env = gym.make("LunarLander-v2")
policy_model = nn_model2((8,), 4)
value_model = nn_model2((8,), 1)
