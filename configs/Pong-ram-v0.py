from utils import nn_model, nn_model2
import gym


config = {
	"correlated_epsilon" : True,
	"epsilon_decay" : lambda x: x - 0.005,
	"gamma" : 0.99,
	"epsilon" : 1.,
	# "n_paths" : 50,
	# "batch_size" : 25000,
	"n_threads" : 2,
}

class ChangeActionSpaceEnv(gym.Wrapper):
	def __init__(self, env):
		"""Take action on reset for environments that are fixed until firing."""
		gym.Wrapper.__init__(self, env)
		self.action_space = gym.spaces.Discrete(2)

	def step(self, ac):
		if ac == 0:
			ac = 2
		else:
			ac = 3
		return self.env.step(ac)

env = gym.make("Pong-ram-v0", difficulty=0, frameskip = 1)
env = ChangeActionSpaceEnv(env)

# if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2', 'Pong-ram-v0']:
policy_model = nn_model2((128,), 2)
value_model = nn_model2((128,), 1)
