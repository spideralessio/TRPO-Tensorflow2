import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def gradient(f, vars):
	with tf.GradientTape(persistent = True) as t:
		f = f()
	grads = t.gradient(f, vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
	return grads

def hessian_matrix(f, vars):
	with tf.GradientTape(persistent = True) as t:
		grads = gradient(f, vars)[0]
		print(grads.shape)
		grads =  tf.transpose(grads)
		print(grads.shape)
	h = t.gradient(grads, vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)[0]
	return tf.convert_to_tensor(h)


def kl_divergence(p, q):
    return tf.reduce_sum(tf.where(p != 0, p * tf.math.log(p / q), 0))


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

def conjugate_grad(A, b, x=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    n = len(b)
    if not x:
        x = np.ones(n)
    r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in xrange(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            break
        p = beta * p - r
    return x