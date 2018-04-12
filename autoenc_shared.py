import tensorflow as tf
import numpy as np
from collections import defaultdict
import itertools
import __main__ as main
from logger import Logger
from data_reader import construct_dataset
from time import sleep

tf.logging.set_verbosity(tf.logging.INFO)

# shared params 
gene_count = 2054
num_hidden_1 = gene_count // 10 # dunno yet
num_input =  gene_count
activation = tf.nn.selu
display_step = 10
threads_num = 2


init_learning_rate = 0.001
num_steps = 100
batch_size = 20
momentum = 0.00001
dropout_p = 0.9
display_step = 10

num_hidden_1 = gene_count // 10 # dunno yet
num_input =  gene_count
activation = tf.nn.selu
step = 0

def wait_step(step):
	sleep(30)
	return tf.get_variable(step, dtype=tf.int64, initializer=tf.zeros_initializer(), shape=[])

def shared_layer(x, cluster_spec):
	with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
	#with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
		weights = tf.get_variable('weights',
			initializer=tf.orthogonal_initializer(),
			shape=[num_hidden_1, num_hidden_1],
			dtype=tf.float64)
		tf.add_to_collection('shared', weights)
		biases = tf.get_variable('biases',
			initializer=tf.zeros_initializer(),
			shape=[num_hidden_1],
			dtype=tf.float64)
		tf.add_to_collection('shared', biases)

	return activation(tf.add(tf.matmul(x, weights), biases))

class Model:
	def __init__(self, cluster, ind, data_filenames, local_batch_size, gene_count, num_steps, batch_count=2):
		import tensorflow as tf 
		self.device = "/job:worker/task:{}".format(ind)
		self.data = construct_dataset(local_batch_size, data_filenames, batch_count)
		self.task_index = ind
		self.num_steps = num_steps
		self.scope_name = 'model{}'.format(ind)
		#self.logger = Logger(self.job)
		with tf.variable_scope(self.scope_name):
			self.x = tf.placeholder(tf.float64, shape=[batch_size, gene_count])
		with tf.variable_scope('steps', reuse=tf.AUTO_REUSE):
			self.step = tf.get_variable('{}/step'.format(self.scope_name), dtype=tf.int64, initializer=tf.zeros_initializer(), shape=[], trainable=False)
		self.server = tf.train.Server(cluster, job_name="worker", task_index=ind)
		self.cluster = cluster
		self.is_chief = (ind == 0)

	def encoder(self, x):
		encoded = tf.contrib.layers.fully_connected(x, num_hidden_1 * 2,
	        activation_fn=activation,
	        weights_initializer=tf.orthogonal_initializer(),
	        biases_initializer=tf.zeros_initializer(),
	        variables_collections=['{}/local_encoder'.format(self.device)])

		encoded_2 = tf.contrib.layers.fully_connected(encoded, num_hidden_1, 
	        activation_fn=activation,
	        weights_initializer=tf.orthogonal_initializer(),
	        biases_initializer=tf.zeros_initializer(),
	        variables_collections=['{}/local_encoder'.format(self.device)])

		encoded_3 = shared_layer(encoded_2, self.cluster)
		return encoded_3

	def decoder(self, x):
		decoded_1 = tf.contrib.layers.fully_connected(x, num_hidden_1, 
	        activation_fn=activation,
	        weights_initializer=tf.orthogonal_initializer(),
	        biases_initializer=tf.zeros_initializer(),
	        variables_collections=['{}/local_decoder'.format(self.device)])


		decoded_2 = tf.contrib.layers.fully_connected(decoded_1, num_hidden_1 * 2, 
	        activation_fn=activation,
	        weights_initializer=tf.orthogonal_initializer(),
	        biases_initializer=tf.zeros_initializer(),
	        variables_collections=['{}/local_decoder'.format(self.device)])

		decoded = tf.contrib.layers.fully_connected(decoded_2, gene_count, 
	        activation_fn=activation,
	        weights_initializer=tf.orthogonal_initializer(),
	        biases_initializer=tf.zeros_initializer(),
	        variables_collections=['{}/local_decoder'.format(self.device)])

		return decoded


	
	def estimate(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)

		self.loss = tf.losses.mean_squared_error(x, decoded)
	
	def collect_losses(self):
		with tf.device(tf.train.replica_device_setter(worker_device=self.device, cluster=self.cluster)):
			with tf.variable_scope('steps', reuse=tf.AUTO_REUSE):
				tf.assign_add(self.step, self.step + 1)		
				steps = ['model{}/step'.format(i) for i in range(threads_num)]
				print(steps)
				step = steps.pop()

				while steps:
					step = tf.cond(tf.equal(tf.get_variable(step, dtype=tf.int64, initializer=tf.zeros_initializer(), shape=[]), self.step), 
						lambda: tf.get_variable(steps.pop(), dtype=tf.int64, initializer=tf.zeros_initializer(), shape=[]),
						lambda: wait_step(step))

			total_loss = tf.get_collection(tf.GraphKeys.LOSSES)
			print(total_loss)

		return tf.reduce_mean(total_loss)


	def update(self, loss):
		local_opt = tf.train.AdamOptimizer()
		print(loss)
		print(tf.get_collection('{}/local_decoder'.format(self.device)))
		grads_vars = local_opt.compute_gradients(loss, var_list=tf.get_collection('{}/local_decoder'.format(self.device)))
		train_op = [local_opt.apply_gradients(grads_vars)]

		with tf.device(tf.train.replica_device_setter(worker_device=self.device, cluster=self.cluster)):
			global_step=tf.train.get_or_create_global_step()
			sync_opt = tf.train.SyncReplicasOptimizer(tf.train.AdamOptimizer(), replicas_to_aggregate=2, total_num_replicas=2)
			train_op.append(sync_opt.minimize(loss, 
				global_step=global_step, 
				var_list=[tf.get_collection('shared'), tf.get_collection('{}/local_encoder'.format(self.device))]))
			sync_replicas_hook = sync_opt.make_session_run_hook(self.is_chief, num_tokens=0)

		grads_vars = local_opt.compute_gradients(loss, var_list=tf.get_collection('{}/local_encoder'.format(self.device)))
		train_op.append(local_opt.apply_gradients(grads_vars))

		return train_op, sync_replicas_hook


	def train(self, x):
		self.estimate(x)

		loss = self.collect_losses()
		return self.update(loss)
	
	def get_batch(self):
		return next(self.data)

	def sess_runner(self):
		train, hook = self.train(self.x)

		with tf.device(tf.train.replica_device_setter(worker_device=self.device, cluster=self.cluster)):
			stop_hook = tf.train.StopAtStepHook(last_step=20)

			with tf.train.MonitoredTrainingSession(master=self.server.target, is_chief=self.is_chief, hooks=[hook, stop_hook]) as sess:
				while not sess.should_stop():
					batch = self.get_batch()

					for el in batch:
						values = {self.x : el}
						for i in range(self.num_steps):						
							print(sess.run([train], feed_dict=values))