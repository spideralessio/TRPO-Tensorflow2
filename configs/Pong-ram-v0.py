from utils import nn_model, nn_model2

config = {
	"correlated_epsilon" : True,
	# "epsilon_decay" : lambda x: x - 1e-3,
	"gamma" : 0.85,
	"epsilon" : 0.4,
	"n_paths" : 20,
}

# if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2', 'Pong-ram-v0']:
policy_model = nn_model2((128,), 6)
value_model = nn_model2((128,), 1)
