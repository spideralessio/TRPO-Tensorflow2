import gym
from utils import nn_model

config = {
	"epsilon_decay" : lambda x: x - 5e-5 
}

env_name = "Pong-ram-v0"

env = gym.make(env_name)

# if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2', 'Pong-ram-v0']:
policy_model = nn_model(env.observation_space.shape, env.action_space.n)
value_model = nn_model(env.observation_space.shape, 1)

env.close()