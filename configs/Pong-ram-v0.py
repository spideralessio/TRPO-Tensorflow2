from utils import nn_model

config = {
	"epsilon_decay" : lambda x: x - 5e-5,
	"reward_scaling" : 0.2,
	"n_paths":30
}

# if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2', 'Pong-ram-v0']:
policy_model = nn_model((128,), 6)
value_model = nn_model((128,), 1)
