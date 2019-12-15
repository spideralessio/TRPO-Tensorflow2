from utils import nn_model
import gym
config = {
	# "correlated_epsilon" : True
}


env = gym.make("Acrobot-v1")
policy_model = nn_model((6,), 3)
value_model = nn_model((6,), 1)
