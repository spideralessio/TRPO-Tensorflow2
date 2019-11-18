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
	def __init__(self, env, lr, gamma, policy_model, epsilon=0, value_model=False, value_lr=2e-2):
		self.env = env
		self.lr = lr
		self.gamma = gamma
		self.epsilon = epsilon
		self.writer = tf.summary.create_file_writer("mylogs/loss")
		self.model = policy_model
		self.optimizer = optimizers.SGD(1)
		self.value_optimizer = optimizers.Adam(value_lr)
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
		actions = np.array(actions)
		return obs, rs, actions

	def train_step(self, episode, obs, rs, actions):

		Gs = []
		G = 0
		for r in rs[::-1]:
			G = r + self.gamma*G
			Gs.insert(0, G)
		Gs = np.array(Gs)
		#Gs = (Gs - Gs.mean())/(Gs.std() + 1e-8)
		if self.value_model:
			with tf.GradientTape() as value_tape:
				Vs = self.value_model(obs)
				advantage = Gs - Vs
				loss = tf.reduce_mean(advantage**2) 
			d_v = value_tape.gradient(loss, self.value_model.trainable_variables)
			self.value_optimizer.apply_gradients(zip(d_v, self.value_model.trainable_variables))
		else:
			advantage = Gs
		with tf.GradientTape() as tape:
			logits = self.model(obs)
			action_prob = tf.nn.softmax(logits)
			actions_one_hot = tf.one_hot(actions, self.env.action_space.n, dtype="float64")
			taken_action_prob = tf.reduce_sum(actions_one_hot * action_prob, axis=1)
			log_prob = tf.math.log(taken_action_prob)
			log_prob_v = tf.negative(tf.reduce_mean(log_prob*advantage))*lr
		d_log_prob = tape.gradient(log_prob_v, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(d_log_prob, self.model.trainable_variables))

		# if self.value_model:
		# 	Vs = self.value_model(obs)
		# 	delta = Gs - Vs # advantage
		# 	loss*=delta
		# loss = tf.reduce_mean(loss * Gs)
		# loss = tf.negative(loss) # negative to maximize strano 
		# return loss
		#loss_fn = lambda: loss
		#loss_fn = lambda: tf.reduce_mean((self.model(obs) - actions_one_hot)**2)
		#loss_fn = lambda: tf.nn.softmax_cross_entropy_with_logits(actions_one_hot, self.model(obs))
		# if self.value_model:
		# 	def value_loss_fn():
		# 		Vs = self.value_model(obs)	
		# 		delta = Gs - Vs 
		# 		value_loss = tf.reduce_mean((Vs - Gs)**2)
		# 		return value_loss
		# 	self.optimizer.minimize(value_loss_fn, self.value_model.trainable_variables)
		# #self.optimizer.minimize(loss_fn, self.model.trainable_variables)
		#total_loss = loss_fn()
		writer = self.writer
		with writer.as_default():
			tf.summary.scalar("reward", rs.sum(), step=episode)
			# # other model code would go here
			# tf.summary.scalar("loss", total_loss, step=episode)
			# if self.value_model:
			# 	total_value_loss = value_loss_fn()
			# 	tf.summary.scalar("value_loss", total_value_loss, step=episode)
		# print(f"Ep {episode}: Rw {rs.sum()} - Loss {total_loss}")
		print(f"Ep {episode}: Rw {rs.sum()}")


	def train(self, episodes):
		for episode in range(episodes):
			
			obs, rs, actions = self.sample()
			total_loss = self.train_step(episode, obs, rs, actions)
			
			
			
			



