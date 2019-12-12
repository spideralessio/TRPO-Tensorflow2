import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gym
from utils import nn_model2
from matplotlib import pyplot as plt
config = {
	"epsilon_decay" : lambda x: max(0.1, x - 1e-3),
	"n_paths" : 20,
	"gamma" : 0.99,
	"batch_size" : 10000,
}


env = gym.make("Pong-v0", difficulty=0, frameskip=1)

policy_model =  nn_model2((210,160,3), 6, convolutional=True)
