import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import numpy as np
class REINFORCE:
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
	def __init__(self, env, lr, gamma, policy_model, epsilon=0, value_model=False):
		self.env = env
		self.lr = lr
		self.gamma = gamma
		self.epsilon = epsilon
		self.writer = tf.summary.create_file_writer("mylogs/loss")
		self.model = policy_model
		self.optimizer = optimizers.Adam(lr)
		self.value_model = value_model
	def __call__(self, ob):
		ob = ob[np.newaxis, :]
		logits = self.model(ob)
		action_prob = tf.nn.softmax(logits)
		action = np.random.choice(range(action_prob.shape[1]), p=action_prob.numpy().ravel())
		
		# epsilon greedy
		if np.random.uniform(0,1) < self.epsilon:
			action = np.random.randint(0,self.env.action_space.n)
		self.epsilon -= self.epsilon*1e-4

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
			#print(ob, action, r, new_ob)
			obs.append(ob)
			rs.append(r)
			actions.append(action)
			ob = new_ob
		obs = np.array(obs)
		rs = np.array(rs)
		return obs, rs, actions

	def train_step(self, episode, obs, rs, actions):

		Gs = []
		G = 0
		for r in rs[::-1]:
			G = r + self.gamma*G
			Gs.insert(0, G)
		Gs = np.array(Gs)
		Gs = (Gs - Gs.mean())/(Gs.std() + 1e-8)
		grads_list = []
		total_value_loss = 0
		with tf.GradientTape() as tape:
			with tf.GradientTape() as value_tape:
				logits = self.model(obs)
				#log_prob = tf.math.log(tf.gather(action_prob, action, axis=1))
				neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=tf.one_hot([1], self.env.action_space.n))
				loss = neg_log_prob * Gs
				#log_prob = tf.math.log(tf.gather(action_prob, action, axis=1))
				#loss = tf.reduce_mean(tf.negative(log_prob*G)) # Negative since we want to maximize and apply_gradient does gradient descent, with minus we do gradient ascent
				if self.value_model:
					Vs = self.value_model(obs)
					delta = Gs - Vs # advantage
					value_loss = tf.losses.mse(V, G)
					loss *= delta
				loss = tf.reduce_mean(loss)
		if self.value_model:
			value_grads = value_tape.gradient(value_loss, self.value_model.trainable_variables)
			self.optimizer.apply_gradients(zip(value_grads, self.value_model.trainable_variables))
		grads = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
			

		writer = self.writer
		with writer.as_default():
			# other model code would go here
			tf.summary.scalar("loss", loss, step=episode)
			if total_value_loss > 0:
				tf.summary.scalar("value_loss", value_loss, step=episode)
			tf.summary.scalar("reward", rs.sum(), step=episode)
		print(f"Ep {episode}: Rw {rs.sum()} - Loss {loss}")
		return loss

	def train(self, episodes):
		for episode in range(episodes):
			
			obs, rs, actions = self.sample()
			total_loss = self.train_step(episode, obs, rs, actions)
			
			
			
			



