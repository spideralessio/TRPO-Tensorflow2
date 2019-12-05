import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, models
import numpy as np
from utils import flatgrad, nn_model, assign_vars, flatvars
import os
import glob
from datetime import datetime
import threading
import gym
import time


class TRPO:
	def __init__(self, env_name, policy_model, value_model=None, value_lr=1e-1, gamma=0.99, delta = 0.01, 
				cg_damping=0.001, cg_iters=10, residual_tol=1e-5, ent_coeff=0.0, epsilon=0.4,
				backtrack_coeff=0.6, backtrack_iters=10, render=False, batch_size=4096, n_paths=10, n_threads=2, epsilon_decay=lambda x: x - 5e-3, reward_scaling = 1., correlated_epsilon=False):
		self.env_name = env_name
		self.N_PATHS = n_paths
		self.N_THREADS = n_threads
		self.envs = []
		self.epsilon_decay = epsilon_decay
		assert self.N_PATHS > 0 and self.N_THREADS > 0
		for i in range(self.N_PATHS):
			self.envs.append(gym.make(self.env_name))
			if self.env_name == "MountainCar-v0":
				self.envs[-1]._max_episode_steps = 1600
		self.gamma = gamma
		self.cg_iters = cg_iters
		self.cg_damping = cg_damping
		self.ent_coeff = ent_coeff
		self.residual_tol = residual_tol
		current_time = datetime.now().strftime('%b%d_%H-%M-%S')
		self.name = f"mylogs/TRPO-{self.env_name}-{current_time}"
		self.model = policy_model
		self.tmp_model = models.clone_model(self.model)
		self.value_model = value_model
		if self.value_model:
			self.value_optimizer = optimizers.Adam(lr=value_lr)
			self.value_model.compile(self.value_optimizer, "mse")
			self.writer = tf.summary.create_file_writer(self.name)
		self.delta = delta
		self.epsilon = epsilon
		self.backtrack_coeff = backtrack_coeff
		self.backtrack_iters = backtrack_iters
		self.render = render
		self.reward_scaling = reward_scaling
		self.correlated_epsilon = correlated_epsilon
		if render:
			os.system("touch render")
		elif not render and len(glob.glob("render")) > 0:
			os.system("rm render")
		self.BATCH_SIZE = batch_size
	def close(self):
		for env in self.envs:
			env.close()
	def __call__(self, ob, last_action=None):
		ob = ob[np.newaxis, :]
		if self.env_name == "Pong-v0":
			ob = tf.image.crop_to_bounding_box(ob, 33,0,160,160)
			ob = tf.cast(tf.image.resize(tf.image.rgb_to_grayscale(ob), size=(32,32))
		logits = self.model(ob)
		action_prob = tf.nn.softmax(logits).numpy().ravel()
		action = np.random.choice(range(action_prob.shape[0]), p=action_prob)
		# epsilon greedy
		if np.random.uniform(0,1) < self.epsilon:
			if self.correlated_epsilon and np.random.uniform(0,1) < 0.8 and last_action is not None:
				action = last_action
			else:
				action = np.random.randint(0,self.envs[0].action_space.n)
				self.last_action = action
		return action, action_prob

	def render_episode(self, n=1):
		for i in range(n):
			ob = self.envs[0].reset()
			done = False
			action = None
			while not done:
				self.envs[0].render()
				action, _ = self(ob, action)
				ob, r, done, info = self.envs[0].step(action)

	def load_weights(self, path):
		self.model.load_weights(path)

	def sample(self, episode):
		obs_all, actions_all, rs_all, action_probs_all, Gs_all = [None]*self.N_PATHS, [None]*self.N_PATHS, [None]*self.N_PATHS, [None]*self.N_PATHS, [None]*self.N_PATHS
		mean_total_reward = [None]*self.N_PATHS
		mean_entropy = [None]*self.N_PATHS
		if len(glob.glob("render")) > 0:
			self.render = True
		else:
			self.render = False

		if self.render:
			self.render_episode()

		def generate_path(path):
			entropy = 0
			obs, actions, rs, action_probs, Gs = [], [], [], [], []
			ob = self.envs[path].reset()
			done = False
			
			last_action = None
			while not done:
				action, action_prob = self(ob, last_action)
				new_ob, r, done, info = self.envs[path].step(action)
				last_action = action
				rs.append(r/self.reward_scaling)
				obs.append(ob)
				actions.append(action)
				action_probs.append(action_prob)
				entropy += - tf.reduce_sum(action_prob*tf.math.log(action_prob))
				ob = new_ob
			G = 0
			for r in rs[::-1]:
				G = r + self.gamma*G
				Gs.insert(0, G)
			mean_total_reward[path] = sum(rs)
			entropy = entropy / len(actions)
			mean_entropy[path] = entropy
			obs_all[path] = obs
			actions_all[path] = actions
			rs_all[path] = rs
			action_probs_all[path] = action_probs
			Gs_all[path] = Gs
		
		i = 0
		while i < self.N_PATHS:
			j = 0
			threads = []
			while j < self.N_THREADS and i < self.N_PATHS:
				thread = threading.Thread(target=generate_path, args=(i,))
				thread.start()
				threads.append(thread)
				j += 1
				i += 1
			for thread in threads:
				thread.join()


		mean_entropy = np.mean(mean_entropy)
		mean_total_reward = np.mean(mean_total_reward)
		Gs_all = np.concatenate(Gs_all)
		obs_all = np.concatenate(obs_all)
		rs_all = np.concatenate(rs_all)
		actions_all = np.concatenate(actions_all)
		action_probs_all = np.concatenate(action_probs_all)
		return obs_all, Gs_all, mean_total_reward, actions_all, action_probs_all, mean_entropy

	def train_step(self, episode, obs_all, Gs_all, actions_all, action_probs_all, total_reward, entropy, t0):
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
			loss = tf.reduce_mean(prob_ratio * advantage) + self.ent_coeff * entropy
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
			return tf.reduce_mean(tf.reduce_sum(old_action_prob * tf.math.log(old_action_prob / action_prob), axis=1))

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
			old_p = p.copy()
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
				old_p = p.copy()
				p = r + beta * p
			return x

		def linesearch(x, fullstep):
			fval = surrogate_loss(x)
			for (_n_backtracks, stepfrac) in enumerate(self.backtrack_coeff**np.arange(self.backtrack_iters)):
				xnew = x + stepfrac * fullstep
				newfval = surrogate_loss(xnew)
				kl_div = kl_fn(xnew)
				if kl_div <= self.delta and newfval >= 0:
					return xnew
				if _n_backtracks == self.backtrack_iters - 1:
					print("Linesearch failed.")
			return x

		print(len(obs_all))
		print(self.model.summary())
		NBATCHES = len(obs_all) // self.BATCH_SIZE 
		if len(obs_all) < self.BATCH_SIZE:
			NBATCHES += 1
		for batch_id in range(NBATCHES):
			obs = obs_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]
			Gs = Gs_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]
			actions = actions_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]
			action_probs = action_probs_all[batch_id*self.BATCH_SIZE: (batch_id + 1)*self.BATCH_SIZE]


			Vs = self.value_model(obs).numpy().flatten()
			# advantage = Gs
			advantage = Gs - Vs
			advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)
			actions_one_hot = tf.one_hot(actions, self.envs[0].action_space.n, dtype="float64")
			policy_loss = surrogate_loss()
			policy_gradient = flatgrad(surrogate_loss, self.model.trainable_variables).numpy()
			



			step_direction = conjugate_grad(hessian_vector_product, policy_gradient)

			shs = .5 * step_direction.dot(hessian_vector_product(step_direction).T)

			lm = np.sqrt(shs / self.delta) + 1e-8
			fullstep = step_direction / lm
			
			oldtheta = flatvars(self.model).numpy()

			theta = linesearch(oldtheta, fullstep)


			if np.isnan(theta).any():
				print("NaN detected. Skipping update...")
			else:
				assign_vars(self.model, theta)

			kl = kl_fn(oldtheta)

			history = self.value_model.fit(obs, Gs, epochs=5, verbose=0)
			value_loss = history.history["loss"][-1]


			print(f"Ep {episode}.{batch_id}: Rw {total_reward} - PL {policy_loss} - VL {value_loss} - KL {kl} - epsilon {self.epsilon} - time {time.time() - t0}")
		if self.value_model:
			writer = self.writer
			with writer.as_default():
				tf.summary.scalar("reward", total_reward, step=episode)
				tf.summary.scalar("value_loss", value_loss, step=episode)
				tf.summary.scalar("policy_loss", policy_loss, step=episode)
		self.epsilon = self.epsilon_decay(self.epsilon)	

	def train(self, episodes):
		assert self.value_model is not None
		print("Starting training, saving checkpoints and logs to:", self.name)
		for episode in range(episodes):
			t0 = time.time()
			obs, Gs, total_reward, actions, action_probs, entropy = self.sample(episode)
			print(f"Sample Time {time.time() - t0}")
			total_loss = self.train_step(episode, obs, Gs, actions, action_probs, total_reward, entropy, t0)
			if episode % 10 == 0 and episode != 0 and self.value_model:
				self.model.save_weights(f"{self.name}/{episode}.ckpt")

			
			
			
			



