from utils import nn_model
import gym

config = {
	# "correlated_epsilon" : True
}


env = gym.make("CartPole-v0")
# if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2', 'Pong-ram-v0']:
policy_model = nn_model((4,), 2)
value_model = nn_model((4,), 1)
