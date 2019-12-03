import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Makes gradient of function loss_fn wrt var_list and
# flattens it to have a 1-D vector 
def flatgrad(loss_fn, var_list):
	with tf.GradientTape() as t:
		loss = loss_fn()
	grads = t.gradient(loss, var_list, unconnected_gradients=tf.UnconnectedGradients.NONE)
	return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

def nn_model(input_shape, output_shape, convolutional=False):
	model = keras.Sequential()
	if convolutional:
		model.add(layers.Lambda(lambda x: tf.cast(tf.image.resize(tf.image.rgb_to_grayscale(x), size=(64,64)), dtype=tf.float64)/256., input_shape=input_shape))
		model.add(layers.Conv2D(10, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((3, 3)))
		model.add(layers.Conv2D(5, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((3, 3)))
		model.add(layers.Flatten())
	# else:
	model.add(layers.Dense(64, input_shape=input_shape, activation='relu'))
	model.add(layers.Dense(64, activation='relu'))
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

def flatvars(model):
	return tf.concat([tf.reshape(v, [-1]) for v in model.trainable_variables], axis=0)

if __name__ == '__main__':
	model = nn_model((3,1), 4)

	model.summary()

	fv = flatvars(model).numpy()

	assign_vars(model, fv)

	fv_new = flatvars(model).numpy()

	print((fv == fv_new).all())
