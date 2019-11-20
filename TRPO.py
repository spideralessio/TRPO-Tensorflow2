import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, models
import numpy as np
from utils import flatgrad, nn_model, assign_vars, flatvars
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
	def __init__(self, env, policy_model, value_model, value_lr=1e-1, gamma=0.95, delta = 0.01, 
				cg_damping=1e-2, cg_iters=10, residual_tol=1e-10, ent_coeff=0.001, 
				backtrack_coeff=0.8, backtrack_iters=10, render=False):
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
		self.value_model = value_model
		self.value_optimizer = optimizers.Adam(value_lr)
		self.delta = delta
		self.epsilon = 0.0
		self.backtrack_coeff = backtrack_coeff
		self.backtrack_iters = backtrack_iters
		self.render = render

	def __call__(self, ob):
		ob = ob[np.newaxis, :]
		logits = self.model(ob)
		action_prob = tf.nn.softmax(logits)
		action = np.random.choice(range(action_prob.shape[1]), p=action_prob.numpy().ravel())
		# epsilon greedy
		if np.random.uniform(0,1) < self.epsilon:
			action = np.random.randint(0,self.env.action_space.n)
		self.epsilon -= 1e-3
		return action, action_prob

	def sample(self, episode):
		entropy = 0
		obs, actions, rs, action_probs = [], [], [], []
		ob = self.env.reset()
		done = False
		while not done:
			action, action_prob = self(ob)
			#self.env.render()
			new_ob, r, done, info = self.env.step(action)
			if self.render and episode > 500:
				env.render()
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
		def policy_loss_fn(model=None):
			if not model:
				model = self.model
			logits = model(obs)
			action_prob = tf.nn.softmax(logits)
			action_prob = tf.reduce_sum(actions_one_hot * action_prob, axis=1)
			old_logits = self.model(obs)
			old_action_prob = tf.nn.softmax(old_logits)
			old_action_prob = tf.reduce_sum(actions_one_hot * old_action_prob, axis=1).numpy() + 1e-8
			prob_ratio = action_prob / old_action_prob # pi(a|s) / pi_old(a|s)
			loss = -tf.reduce_mean(prob_ratio * advantage) - self.ent_coeff * entropy
			return loss

		def kl_fn(model=None):
			if not model:
				model = self.model
			logits = model(obs)
			action_prob = tf.nn.softmax(logits).numpy()
			old_logits = self.model(obs)
			old_action_prob = tf.nn.softmax(old_logits)
			return tf.reduce_mean(tf.reduce_sum(old_action_prob * tf.math.log(action_prob / (old_action_prob + 1e-8)), axis=1))

		def hessian_vector_product(p):
			def hvp_fn(): 
				kl_grad_vector = flatgrad(kl_fn, self.model.trainable_variables)
				grad_vector_product = tf.reduce_sum(kl_grad_vector * p)
				return grad_vector_product

			fisher_vector_product = flatgrad(hvp_fn, self.model.trainable_variables).numpy()
			return fisher_vector_product + (self.cg_damping * p)

		def conjugate_grad(Ax, b):
			"""
			Conjugate gradient algorithm
			(see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
			"""
			x = np.zeros_like(b)
			r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
			p = r.copy()
			r_dot_old = np.dot(r,r)
			for _ in range(self.cg_iters):
				z = Ax(p)
				alpha = r_dot_old / (np.dot(p, z) + 1e-8)
				x += alpha * p
				r -= alpha * z
				r_dot_new = np.dot(r,r)
				beta = r_dot_new / (r_dot_old + 1e-8)
				r_dot_old = r_dot_new
				if r_dot_old < self.residual_tol:
					break
				p = r + beta * p
			return x
		policy_loss = policy_loss_fn()
		g = flatgrad(policy_loss_fn, self.model.trainable_variables).numpy()
		Hx = hessian_vector_product(g)
		x = conjugate_grad(hessian_vector_product, g)
		alpha = np.sqrt(2*self.delta/(np.dot(x.T, Hx)+1e-8)) * x
		theta = flatvars(self.model).numpy()
		for j in range(self.backtrack_iters):
			theta_new = theta + alpha * self.backtrack_coeff**j
			assign_vars(self.tmp_model, theta_new)
			new_policy_loss = policy_loss_fn(self.tmp_model)
			kl = kl_fn(self.tmp_model)
			if kl <= self.delta and new_policy_loss <= policy_loss:
				theta = theta_new
				break
			if j==self.backtrack_iters-1:
				print('Line search failed! Keeping old params.')
				kl = kl_fn(self.model)
		
		if any(np.isnan(theta)):
			print("NaN detected. Skipping update...")
		else:
			assign_vars(self.model, theta)
		with tf.GradientTape() as value_tape:
			Vs = self.value_model(obs)
			advantage = Gs - Vs
			value_loss = tf.reduce_mean(advantage**2) 
		
		d_v = value_tape.gradient(value_loss, self.value_model.trainable_variables)
		self.value_optimizer.apply_gradients(zip(d_v, self.value_model.trainable_variables))
		

		writer = self.writer
		with writer.as_default():
			tf.summary.scalar("reward", total_reward, step=episode)
			tf.summary.scalar("value_loss", value_loss, step=episode)
			tf.summary.scalar("policy_loss", policy_loss, step=episode)
		print(f"Ep {episode}: Rw {total_reward} - PL {policy_loss} - VL {value_loss} - KL {kl}")


	def train(self, episodes):
		for episode in range(episodes):
			obs, Gs, total_reward, actions, action_probs, entropy = self.sample(episode)
			total_loss = self.train_step(episode, obs, Gs, total_reward, actions, action_probs, entropy)
			
			
			
			



