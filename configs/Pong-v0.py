from utils import nn_model

config = {
	"epsilon_decay" : lambda x: x - 5e-5 
}

policy_model = nn_model((210,160,3), 6, convolutional=True)
value_model = nn_model((210,160,3), 1, convolutional=True)
