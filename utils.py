import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Makes gradient of function loss_fn wrt var_list and
# flattens it to have a 1-D vector 
def flatgrad(loss_fn, var_list):
	with tf.GradientTape() as t:
		loss = loss_fn()
	grads = t.gradient(loss, var_list)
	return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

def nn_model(input_shape, output_shape, convolutional=False):
	model = keras.Sequential()
	if convolutional:
		model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
		#model.add(layers.MaxPooling2D((2, 2)))
		#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.Flatten())
	else:
		model.add(layers.Dense(10, input_shape=input_shape, activation='relu'))
	model.add(layers.Dense(20, activation='relu'))
	model.add(layers.Dense(output_shape))
	return model

def assign_vars(model, theta):
        """
        Create the process of assigning updated vars
        """
        shapes = [v.shape.as_list() for v in model.trainable_variables]	
        size_theta = np.sum([np.prod(shape) for shape in shapes])

        # self.assign_weights_op = tf.assign(self.flat_weights, self.flat_wieghts_ph)
        start = 0
        for i, shape in enumerate(shapes):
            size = np.prod(shape)
            param = tf.reshape(theta[start:start + size], shape)
            model.trainable_variables[i].assign(param)
            start += size
        assert start == size_theta, "messy shapes"