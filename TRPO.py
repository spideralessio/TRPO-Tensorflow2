import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, models
import numpy as np
from utils import flatgrad, nn_model, assign_vars
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
	def __init__(self, env, policy_model, value_model, value_lr=2e-2, gamma=0.99, delta = 0.001, cg_damping=1e-2, cg_iters=10, residual_tol=1e-10, ent_coeff=0.001):
		self.env = env
		self.gamma = gamma
		self.cg_iters = cg_iters
		self.cg_damping = cg_damping
		self.ent_coeff = ent_coeff
		self.residual_tol = residual_tol
		os.system("rm -rf mylogs/trpo")
		self.writer = tf.summary.create_file_writer("mylogs/trpo")
		self.model = policy_model
		self.tmp_model = models.clone_model(self.model)
		self.old_model = models.clone_model(self.model)
		self.old_model.set_weights(self.model.get_weights())
		self.value_model = value_model
		self.optimizer = optimizers.SGD(1)
		self.value_optimizer = optimizers.Adam(value_lr)
		self.delta = delta
		self.epsilon = 0.0

	def __call__(self, ob):
		ob = ob[np.newaxis, :]
		logits = self.model(ob)
		action_prob = tf.nn.softmax(logits)
		action = np.random.choice(range(action_prob.shape[1]), p=action_prob.numpy().ravel())
		# epsilon greedy
		# if np.random.uniform(0,1) < self.epsilon:
		# 	action = np.random.randint(0,self.env.action_space.n)
		# self.epsilon -= self.epsilon*1e-4
		return action, action_prob

	def sample(self):
		entropy = 0
		obs, actions, rs, action_probs = [], [], [], []
		ob = self.env.reset()
		done = False
		while not done:
			action, action_prob = self(ob)
			#self.env.render()
			new_ob, r, done, info = self.env.step(action)
			#print(ob, action, r, new_ob)
			obs.append(ob)
			rs.append(r)
			actions.append(action)
			action_probs.append(action_prob)
			entropy += - tf.reduce_sum(action_prob*tf.math.log(action_prob))
			ob = new_ob
		obs = np.array(obs)
		rs = np.array(rs)
		actions = np.array(actions)
		Gs = []
		G = 0
		for r in rs[::-1]:
			G = r + self.gamma*G
			Gs.insert(0, G)
		Gs = np.array(Gs)
		total_reward = rs.sum()
		entropy = entropy / len(actions)
		return obs, Gs, total_reward, actions, action_probs, entropy

	def train_step(self, episode, obs, Gs, total_reward, actions, action_probs, entropy):

		
		Vs = self.value_model(obs).numpy().flatten()
		advantage = Gs - Vs
		advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)
		actions_one_hot = tf.one_hot(actions, self.env.action_space.n, dtype="float64")

		def g_fn(model=None):
			if not model:
				model = self.model
				old_model = self.old_model
			else:
				old_model = model
			logits = model(obs)
			action_prob = tf.nn.softmax(logits)
			action_prob = tf.reduce_sum(actions_one_hot * action_prob, axis=1)
			old_logits = old_model(obs)
			old_action_prob = tf.nn.softmax(logits)
			old_action_prob = tf.reduce_sum(actions_one_hot * old_action_prob, axis=1) + 1e-8
			# old_action_prob = action_prob.numpy() + 1e-8
			prob_ratio = action_prob / old_action_prob
			loss = tf.reduce_mean(tf.math.multiply(prob_ratio, advantage)) - self.ent_coeff * entropy
			return loss

		def kl_fn():
			logits = self.model(obs)
			action_prob = tf.nn.softmax(logits)
			action_prob = tf.reduce_sum(actions_one_hot * action_prob, axis=1) + 1e-8
			old_logits = self.old_model(obs)
			old_action_prob = tf.nn.softmax(logits)
			old_action_prob = tf.reduce_sum(actions_one_hot * old_action_prob, axis=1)
			# old_action_prob = action_prob.numpy() + 1e-8
			return tf.reduce_mean(old_action_prob * tf.math.log(old_action_prob/action_prob))
			# return tf.reduce_mean(old_action_prob * tf.math.log(old_action_prob/action_prob))

		def hessian_vector_product(p):
			shapes = [v.shape.as_list() for v in self.model.trainable_variables]
			size_theta = np.sum([np.prod(shape) for shape in shapes])

			def hvp_fn(): 
				kl_grad_vector = flatgrad(kl_fn, self.model.trainable_variables)
				grad_vector_product = tf.reduce_sum(kl_grad_vector * p)
				return grad_vector_product

			fisher_vector_product = flatgrad(hvp_fn, self.model.trainable_variables).numpy()
			return fisher_vector_product + (self.cg_damping * p)


				# with tf.GradientTape() as t:
				# 	kl_div = kl_fn()

				# grads = t.gradient(kl_div, self.model.trainable_variables)
				
				# tangents = []
				# start = 0
				# for shape in shapes:
				# 	size = np.prod(shape)
				# 	tangents.append(tf.reshape(p[start:start + size], shape))
				# 	start += size
				# gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zip(grads, tangents)])
				# return gvp
			# return flatgrad(hvp_fn, self.model.trainable_variables) * self.cg_damping * p
		def conjugate_grad(b):
			r = b
			p = b
			x = np.zeros_like(b)
			r_k_norm = np.dot(r, r)
			for i in range(self.cg_iters):
				Ap = hessian_vector_product(p)
				alpha = r_k_norm / np.dot(p, Ap)
				x += alpha * p
				r -= alpha * Ap
				r_kplus1_norm = np.dot(r, r)
				beta = r_kplus1_norm / r_k_norm
				r_k_norm = r_kplus1_norm
				if r_kplus1_norm < self.residual_tol:
					break
				p = beta * p + r
			return x
		g = flatgrad(g_fn, self.model.trainable_variables).numpy()
		if g.nonzero()[0].size:
			step_direction = conjugate_grad(-g)
			shs = .5 * step_direction.dot(hessian_vector_product(step_direction).T)
			lm = np.sqrt(shs / self.delta)
			fullstep = step_direction / lm
			gdotstepdir = -g.dot(step_direction)
			expected_improve_rate = gdotstepdir/lm

			theta = tf.concat([tf.reshape(v, [-1]) for v in self.model.trainable_variables], axis=0).numpy()

			accept_ratio = .1
			max_backtracks = 10
			fval = g_fn()
			
			for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
				theta_new = theta + stepfrac * fullstep
				assign_vars(self.tmp_model, theta_new)
				newfval = g_fn(self.tmp_model)
				actual_improve = fval - newfval
				expected_improve = expected_improve_rate * stepfrac
				ratio = actual_improve / expected_improve
				if ratio > accept_ratio and actual_improve > 0:
					theta = theta_new
					break
			
			if any(np.isnan(theta)):
				print("NaN detected. Skipping update...")
			else:
				self.old_model.set_weights(self.model.get_weights())
				assign_vars(self.model, theta)
			with tf.GradientTape() as value_tape:
				Vs = self.value_model(obs)
				advantage = Gs - Vs
				loss = tf.reduce_mean(advantage**2) 
			d_v = value_tape.gradient(loss, self.value_model.trainable_variables)
			self.value_optimizer.apply_gradients(zip(d_v, self.value_model.trainable_variables))
			

			writer = self.writer
			with writer.as_default():
				tf.summary.scalar("reward", total_reward, step=episode)
			print(f"Ep {episode}: Rw {total_reward}")
		else:
			print("Policy gradient is 0. Skipping update...")



	def train(self, episodes):
		for episode in range(episodes):
			obs, Gs, total_reward, actions, action_probs, entropy = self.sample()
			total_loss = self.train_step(episode, obs, Gs, total_reward, actions, action_probs, entropy)
			
			
			
			



