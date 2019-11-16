import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import numpy as np
N  = 1000                           
n  = 4                                  
X = tf.Variable(np.random.randn(n, 1))  
C = tf.constant(np.random.randn(N, n)) 
D = tf.constant(np.random.randn(N, 1))

def var():
	return X
f_batch_tensorflow = lambda: tf.reduce_sum(tf.square(tf.matmul(C, X) - D))
print(f_batch_tensorflow)
print(f_batch_tensorflow())
optimizer = tf.keras.optimizers.Adam().minimize(f_batch_tensorflow, X)