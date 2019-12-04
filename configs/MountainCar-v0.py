from utils import nn_model

config = {}

# if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2', 'Pong-ram-v0']:
policy_model = nn_model((2,), 3)
value_model = nn_model((2,), 1)
