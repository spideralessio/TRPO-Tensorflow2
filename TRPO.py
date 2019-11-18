import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import numpy as np
from utils import gradient, hessian_matrix, kl_divergence, conjugate_grad
import os
class TRPO:
	"""
	REINFORCE Policy Gradient Agent

	...

	Attributes
	----------
	env : OpenAI Gym Environment
		the environment on which the agent is trained and evaluated
	lr : float
		the learning rate
	gamma : float
		reward discounting factor

	Methods
	-------
	says(sound=None)
		Prints the animals name and what sound it makes
	"""
	def __init__(self, env, policy_model, value_model, lr=2e-2, value_lr=2e-2, gamma=0.95, delta = 0.5):
		self.env = env
		self.lr = lr
		self.gamma = gamma
		os.system("rm -rf mylogs/trpo")
		self.writer = tf.summary.create_file_writer("mylogs/trpo")
		self.model = policy_model
		self.old_model = keras.models.clone_model(self.model)
		self.optimizer = optimizers.SGD(1)
		self.value_optimizer = optimizers.Adam(value_lr)
		self.value_model = value_model
		self.delta = delta

	def __call__(self, ob):
		ob = ob[np.newaxis, :]
		logits = self.model(ob)
		action_prob = tf.nn.softmax(logits)
		action = np.random.choice(range(action_prob.shape[1]), p=action_prob.numpy().ravel())
		return action, action_prob

	def sample(self):
		obs = []
		rs = []
		actions = []
		ob = self.env.reset()
		done = False
		while not done:
			action, action_prob = self(ob)
			new_ob, r, done, info = self.env.step(action)
			obs.append(ob)
			rs.append(r)
			actions.append(action)
			ob = new_ob
		obs = np.array(obs)
		rs = np.array(rs)
		actions = np.array(actions)
		return obs, rs, actions

	def train_step(self, episode, obs, rs, actions):

		Gs = []
		G = 0
		for r in rs[::-1]:
			G = r + self.gamma*G
			Gs.insert(0, G)
		Gs = np.array(Gs)
		Vs = self.value_model(obs)
		advantage = Gs - Vs
		#Gs = (Gs - Gs.mean())/(Gs.std() + 1e-8)
		
		def g_fn():
			logits = self.model(obs)
			action_prob = tf.nn.softmax(logits)
			actions_one_hot = tf.one_hot(actions, self.env.action_space.n, dtype="float64")
			taken_action_prob = tf.reduce_sum(actions_one_hot * action_prob, axis=1)
			log_prob = tf.math.log(taken_action_prob)
			return tf.reduce_mean(log_prob*advantage)

		def H_fn():
			return kl_divergence(tf.nn.softmax(self.old_model(obs)), tf.nn.softmax(self.model(obs)))

		print(H_fn())

		for v in self.model.trainable_variables:
			print(v.shape)
			H = hessian_matrix(H_fn, [v])
			g = gradient(g_fn, [v])[0]
			print(H.shape)
			print(g.shape)
			exit(1)


		# gs = gradient(g_fn, self.model.trainable_variables)
		# print(gs)
		# g = tf.reshape(tf.convert_to_tensor(gs), [-1,1])
		
		# H = hessian(H_fn, self.model.trainable_variables)

		# x = conjugate_grad(H.numpy(), g.numpy())

		# DELTA = tf.reshape(tf.math.sqrt(2*self.delta/(tf.transpose(x)*H*x))*x, [-1]).numpy.to_list()

		# Line Search
		# j = 0
		# while True:
		# 	tmp_model = keras.models.clone_model(self.model)
		# 	tmp_model.set_weights(self.model.get_weights())
		# 	grads = [grad * lr**j for grad in DELTA]
		# 	self.optimizer.apply_gradients(zip(grads, tmp_model.trainable_variables))
		# 	if qualcosa and kl_divergence(tf.nn.softmax(self.model(obs)), tf.nn.softmax(tmp_model(obs))) < self.delta:
		# 		self.old_model = self.model
		# 		self.model = tmp_model
		# 		break

		with tf.GradientTape() as value_tape:
			Vs = self.value_model(obs)
			advantage = Gs - Vs
			loss = tf.reduce_mean(advantage**2) 
		d_v = value_tape.gradient(loss, self.value_model.trainable_variables)
		self.value_optimizer.apply_gradients(zip(d_v, self.value_model.trainable_variables))
		

		writer = self.writer
		with writer.as_default():
			tf.summary.scalar("reward", rs.sum(), step=episode)
		print(f"Ep {episode}: Rw {rs.sum()}")


	def train(self, episodes):
		for episode in range(episodes):
			obs, rs, actions = self.sample()
			total_loss = self.train_step(episode, obs, rs, actions)
			
			
			
			



