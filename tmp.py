import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import numpy as np

opt = tf.keras.optimizers.SGD(learning_rate=0.1)


def get_model(w = None):
	model = keras.Sequential()
	model.add(layers.Dense(10, activation='relu', input_shape=(1,)))
	model.add(layers.Dense(1))
	if w:
		model.set_weights(w)
	return model
model = get_model()
x = np.array([[1]])
y = np.array([[30]])

print("model", model(x))

w = model.get_weights()

model1 = get_model(w)
print("model1", model1(x))
for i in range(30):
	loss_fn = lambda: tf.reduce_mean(tf.math.log(model1(x) - y))
	var_list_fn = lambda: model1.trainable_variables
	def compute_loss():
	    return tf.reduce_mean((model1(x) - y)**2)
	opt.minimize(loss_fn, var_list=model1.trainable_variables)


model2 = get_model(w)
print("model2", model2(x))
for i in range(30):
	with tf.GradientTape() as tape:
		#loss = tf.losses.mse(model2(x), y)
		loss = tf.reduce_mean((model2(x) - y)**2)


		#loss = y - model2(x)
	grads = tape.gradient(loss, model2.trainable_variables)
	opt.apply_gradients(zip(grads,model2.trainable_variables))

print("model1", model1(x))
print("model2", model2(x))


print(w)