import tensorflow as tf

x = tf.Variable([[1.],[1.]])
y = tf.Variable([[2.],[2.]])
z = tf.Variable([[3.],[3.]])
f = lambda: (x**3)*(z) + (y)*(z**2)

def gradient(f, vars):
	with tf.GradientTape() as t:
		f = f()
	grads = t.gradient(f, vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
	return grads

def hessian_matrix(f, vars):
	with tf.GradientTape(persistent = True) as t:
		grads = gradient(f, vars)
		grad = grads[0]
	print(grad)
	h = t.gradient(grad, vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)[0]
	return h


print(hessian_matrix(f, [x]))