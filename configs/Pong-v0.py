import gym
from utils import nn_model

config = {
	"epsilon_decay" : lambda x: x - 5e-5 
}

env_name = "Pong-v0"

env = gym.make(env_name)

policy_model = nn_model(env.observation_space.shape, env.action_space.n, convolutional=True)
value_model = nn_model(env.observation_space.shape, 1, convolutional=True)

env.close()