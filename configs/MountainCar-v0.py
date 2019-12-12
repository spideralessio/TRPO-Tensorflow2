from utils import nn_model
import gym

config = {
	"correlated_epsilon" : True
}
gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=1600,      # MountainCar-v0 uses 200
    # reward_threshold=-110.0,
)
env = gym.make("MountainCarMyEasyVersion-v0")
env.max_episode_steps = 1600

# if env_name in ['MountainCar-v0', 'CartPole-v0', 'Acrobot-v1', 'LunarLander-v2', 'Pong-ram-v0']:
policy_model = nn_model((2,), 3)
value_model = nn_model((2,), 1)
