import tensorflow as tf
import numpy as np
from logger import Logger
from data_reader import Dataset

tf.logging.set_verbosity(tf.logging.INFO)

# shared params 
gene_count = 2054
num_hidden_1 = gene_count // 10 # dunno yet
num_input =  gene_count
activation = tf.nn.selu

init_learning_rate = 0.001
display_step = 10


def shared_layer(x):
	"""encoder layer shared across all models"""
	
	with tf.variable_scope('shared',
		reuse=tf.AUTO_REUSE):

		weights = tf.get_variable('weights',
			initializer=tf.orthogonal_initializer(),
			shape=[num_hidden_1, num_hidden_1],
			dtype=tf.float64)

		biases = tf.get_variable('biases',
			initializer=tf.zeros_initializer(),
			shape=[num_hidden_1],
			dtype=tf.float64)

	return activation(tf.add(tf.matmul(x, weights), biases))

class Model:
	""" class representing model for one dataset """
	def __init__(self,
		dataset, 
		delay_step=1,
		ind=0):
		"""
		dataset -- Dataset instance containing dataset to be fed to model
		delay_step -- how many global training steps it takes to run one training iter of model
		ind -- unique index of model
		"""

		self.data = dataset
		self.task_index = ind
		self.scope_name = 'model{}'.format(ind)

		with tf.variable_scope(self.scope_name):

			self.x = tf.placeholder(tf.float64,
				shape=[batch_size, dataset.count()])

			self.delay_step = tf.constant(delay_step,
				name='delay_step')

	def encoder(self, x):
		"""define local encoder"""

		scope_name = '{}/local_encoder'.format(self.scope_name)

		encoded = tf.contrib.layers.fully_connected(x,
			num_hidden_1 * 2,
	        activation_fn=activation,
	        weights_initializer=tf.orthogonal_initializer(),
	        biases_initializer=tf.zeros_initializer(),
	        variables_collections=[scope_name])

		encoded_2 = tf.contrib.layers.fully_connected(encoded,
			num_hidden_1, 
	        activation_fn=activation,
	        weights_initializer=tf.orthogonal_initializer(),
	        biases_initializer=tf.zeros_initializer(),
	        variables_collections=[scope_name])

		#shared layer
		encoded_3 = shared_layer(encoded_2, self.cluster)
		return encoded_3

	def decoder(self, x):
		"""define local decoder"""

		scope_name = '{}/local_decoder'.format(self.scope_name)

		decoded_1 = tf.contrib.layers.fully_connected(x,
			num_hidden_1, 
	        activation_fn=activation,
	        weights_initializer=tf.orthogonal_initializer(),
	        biases_initializer=tf.zeros_initializer(),
	        variables_collections=[scope_name])


		decoded_2 = tf.contrib.layers.fully_connected(decoded_1,
			num_hidden_1 * 2, 
	        activation_fn=activation,
	        weights_initializer=tf.orthogonal_initializer(),
	        biases_initializer=tf.zeros_initializer(),
	        variables_collections=[scope_name])

		decoded = tf.contrib.layers.fully_connected(decoded_2,
			self.dataset.count(), 
	        activation_fn=activation,
	        weights_initializer=tf.orthogonal_initializer(),
	        biases_initializer=tf.zeros_initializer(),
	        variables_collections=[scope_name])

		return decoded


	def estimate(self):
		"""append model's ops to graph"""

		encoded = self.encoder(self.x)
		decoded = self.decoder(encoded)

		return tf.losses.mean_squared_error(self.x, decoded)

	def collect_loss(self, global_step):
		"""
		collect loss according to global step
		
		global_step -- tf.Tensor containing global session step
		"""

		should_run = tf.equal(global_step, self.delay_step)
		loss = tf.cond(should_run,
			estimate(),
			tf.to_float(0.0))

		return loss


	def get_batch(self):
		return next(self.data)
	

def sess_runner(sets, cluster):
	global_step = tf.train.create_global_step()
	losses = [m.collect_loss(global_step) for m in sets]
	total_loss = tf.reduce_mean(losses)
	opt = tf.train.AdamOptimizer()
	train_op = opt.minimize(total_loss)
	

	stop_hook = tf.train.StopAtStepHook(last_step=20)

	with tf.train.MonitoredTrainingSession(hooks=[stop_hook],
		config=tf.ConfigProto(log_device_placement=True)) as sess:

		for i in range(iter_count):
			for s in sets:
				if not i % s.delay_step:
					values[s.x] = s.get_batch()

			for i in range(200):						
				print(sess.run([total_loss, train_op], feed_dict=values))