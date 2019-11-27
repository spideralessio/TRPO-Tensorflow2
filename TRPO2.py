import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, models
import numpy as np
from utils import flatgrad, nn_model, assign_vars, flatvars
import os
import glob
from datetime import datetime
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
	def __init__(self, env, policy_model, value_model, value_lr=1e-1, gamma=0.99, delta = 0.01, 
				cg_damping=0.001, cg_iters=10, residual_tol=1e-5, ent_coeff=0.001, epsilon=0.0,
				backtrack_coeff=0.6, backtrack_iters=15, render=False, batch_size=64, n_paths=10):
		self.env = env
		self.gamma = gamma
		self.cg_iters = cg_iters
		self.cg_damping = cg_damping
		self.ent_coeff = ent_coeff
		self.residual_tol = residual_tol
		#os.system("rm -rf mylogs/trpo")
		current_time = datetime.now().strftime('%b%d_%H-%M-%S')
		self.writer = tf.summary.create_file_writer(f"mylogs/TRPO-{self.env.unwrapped.spec.id}-{current_time}")
		self.model = policy_model
		self.tmp_model = models.clone_model(self.model)
		self.value_model = value_model
		self.value_optimizer = optimizers.Adam(lr=value_lr)
		self.value_model.compile(self.value_optimizer, "mse")
		self.delta = delta
		self.epsilon = epsilon
		self.backtrack_coeff = backtrack_coeff
		self.backtrack_iters = backtrack_iters
		self.render = render
		if render:
			os.system("touch render")
		self.N_PATHS = n_paths
		self.BATCH_SIZE = batch_size

	def __call__(self, ob):
		ob = ob[np.newaxis, :]
		logits = self.model(ob)
		action_prob = tf.nn.softmax(logits).numpy().ravel()
		action = np.random.choice(range(action_prob.shape[0]), p=action_prob)
		# epsilon greedy
		if np.random.uniform(0,1) < self.epsilon:
			action = np.random.randint(0,self.env.action_space.n)
		return action, action_prob

	def sample(self, episode):
		entropy = 0
		obs_all, actions_all, rs_all, action_probs_all, Gs_all = [], [], [], [], []
		mean_total_reward = []
		mean_entropy = []
		for path in range(self.N_PATHS):
			obs, actions, rs, action_probs, Gs = [], [], [], [], []
			ob = self.env.reset()
			done = False
			if len(glob.glob("render")) > 0:
				self.render = True
			else:
				self.render = False
			while not done:
				action, action_prob = self(ob)
				#self.env.render()
				new_ob, r, done, info = self.env.step(action)
				if self.render:
					self.env.render()
				rs.append(r)
				obs.append(ob)
				actions.append(action)
				action_probs.append(action_prob)
				entropy += - tf.reduce_sum(action_prob*tf.math.log(action_prob))
				ob = new_ob
			G = 0
			for r in rs[::-1]:
				G = r + self.gamma*G
				Gs.insert(0, G)
			mean_total_reward.append(sum(rs))
			entropy = entropy / len(actions)
			mean_entropy.append(entropy)
			obs_all.extend(obs)
			actions_all.extend(actions)
			rs_all.extend(rs)
			action_probs_all.extend(action_probs)
			Gs_all.extend(Gs)
		mean_entropy = np.mean(mean_entropy)
		mean_total_reward = np.mean(mean_total_reward)
		Gs_all = np.array(Gs_all)
		obs_all = np.array(obs_all)
		rs_all = np.array(rs_all)
		actions_all = np.array(actions_all)
		action_probs_all = np.array(action_probs_all)
		return obs_all, Gs_all, mean_total_reward, actions_all, action_probs_all, mean_entropy

	def train_step(self, episode, obs_all, Gs_all, actions_all, action_probs_all, total_reward, entropy):
		def surrogate_loss(theta=None):
			if theta is None:
				model = self.model
			else:
				model = self.tmp_model
				assign_vars(self.tmp_model, theta)
			logits = model(obs)
			action_prob = tf.nn.softmax(logits)
			action_prob = tf.reduce_sum(actions_one_hot * action_prob, axis=1)
			old_logits = self.model(obs)
			old_action_prob = tf.nn.softmax(old_logits)
			old_action_prob = tf.reduce_sum(actions_one_hot * old_action_prob, axis=1).numpy() + 1e-8
			prob_ratio = action_prob / old_action_prob # pi(a|s) / pi_old(a|s)
			loss = -tf.reduce_mean(prob_ratio * advantage) - self.ent_coeff * entropy
			return loss

		def kl_fn(theta=None):
			if theta is None:
				model = self.model
			else:
				model = self.tmp_model
				assign_vars(self.tmp_model, theta)
			logits = model(obs)
			action_prob = tf.nn.softmax(logits).numpy() + 1e-8
			old_logits = self.model(obs)
			old_action_prob = tf.nn.softmax(old_logits)
			return tf.reduce_mean(tf.reduce_sum(old_action_prob * tf.math.log(old_action_prob / (action_prob)), axis=1))

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

		def linesearch(x, fullstep, expected_improve_rate):
			accept_ratio = .1
			fval = surrogate_loss(x)
			for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(self.backtrack_iters)):
				xnew = x + stepfrac * fullstep
				newfval = surrogate_loss(xnew)
				actual_improve = fval - newfval
				expected_improve = expected_improve_rate * stepfrac
				ratio = actual_improve / expected_improve
				if ratio > accept_ratio and actual_improve >= 0:
					return xnew
				if _n_backtracks == self.backtrack_iters - 1:
					print(ratio, accept_ratio)
					print(actual_improve)
					print("Linesearch failed.")
			return x


		NBATCHES = len(obs_all) // self.BATCH_SIZE 
		if len(obs_all) < self.BATCH_SIZE:
			NBATCHES += 1
		for batch_id in range(NBATCHES):
			obs = obs_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]
			Gs = Gs_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]
			actions = actions_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]
			action_probs = action_probs_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]


			Vs = self.value_model(obs).numpy().flatten()
			advantage = Gs - Vs
			advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)
			actions_one_hot = tf.one_hot(actions, self.env.action_space.n, dtype="float64")
			policy_loss = surrogate_loss()
			if np.isnan(policy_loss):
				print("policy loss is nan")
				print("actions", np.isnan(actions).any())
				print("obs", np.isnan(obs).any())
				print("Gs", np.isnan(Gs).any())
				print("adv", np.isnan(advantage).any())
			policy_gradient = flatgrad(surrogate_loss, self.model.trainable_variables).numpy()
			step_direction = conjugate_grad(hessian_vector_product, -policy_gradient)
			shs = .5 * step_direction.dot(hessian_vector_product(step_direction).T)
			lm = np.sqrt(shs / self.delta) + 1e-8
			fullstep = step_direction / lm
			gdotstepdir = -policy_gradient.dot(step_direction)

			oldtheta = flatvars(self.model).numpy()

			theta = linesearch(oldtheta, fullstep, gdotstepdir/lm)


			if np.isnan(theta).any():
				print("NaN detected. Skipping update...")
			else:
				assign_vars(self.model, theta)



			kl = kl_fn(oldtheta)

			
			
			history = self.value_model.fit(obs, Gs, epochs=5, verbose=0)
			value_loss = history.history["loss"][-1]


			print(f"Ep {episode}.{batch_id}: Rw {total_reward} - PL {policy_loss} - VL {value_loss} - KL {kl}")

		writer = self.writer
		with writer.as_default():
			tf.summary.scalar("reward", total_reward, step=episode)
			tf.summary.scalar("value_loss", value_loss, step=episode)
			tf.summary.scalar("policy_loss", policy_loss, step=episode)
		self.epsilon -= 1e-3	

	def train(self, episodes):
		for episode in range(episodes):
			obs, Gs, total_reward, actions, action_probs, entropy = self.sample(episode)
			total_loss = self.train_step(episode, obs, Gs, actions, action_probs, total_reward, entropy)
			
			
			
			



