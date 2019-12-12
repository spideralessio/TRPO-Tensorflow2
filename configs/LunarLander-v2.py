from utils import nn_model2

config = {
	"correlated_epsilon" : True,
	"epsilon_decay" : lambda x: x,
	"gamma" : 0.9,
	"epsilon" : 0.4,
	"n_paths" : 15,
}

# if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2', 'Pong-ram-v0']:
policy_model = nn_model2((8,), 4)
value_model = nn_model2((8,), 1)
