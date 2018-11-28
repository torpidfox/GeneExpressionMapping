from data_reader_r import Data
import tensorflow as tf
import numpy as np 

init = tf.contrib.layers.xavier_initializer()

shared_scope = 'shared'
gene_count = 2053
num_hidden_2 = gene_count // 3
num_hidden_1 = gene_count // 2
batch_size = 50
shared_shape = [num_hidden_1, num_hidden_2, num_hidden_2, num_hidden_2]
activation = tf.nn.selu

init_weights = lambda n1, n2: tf.Variable(
            tf.random_normal([n1, n2], 0, np.sqrt(2 / n1)))
init_zeros = lambda n1: tf.Variable([0] * n1, dtype = 'float')

layer = lambda x, v: tf.nn.xw_plus_b(x, v['w'], v['b'])

recon_loss = lambda x1, x2: tf.losses.mean_squared_error(x1, x2)

def init_variables(shape):
	variable_list = list()

	for i, dim in enumerate(shape[:-1]):
		l = {'w' : init_weights(dim, shape[i + 1]),
		'b' : init_zeros(shape[i + 1])}
		variable_list.append(l)

	return variable_list

def nn(layers, x, is_enc=False, is_private=True):
	for i, l in enumerate(layers):
		if i != len(layers) - 1:
			x = activation(layer(x, l))
			#x = tf.nn.dropout(x, 0.8) if i == 0 and is_enc else x
		elif not is_enc and is_private:
			x = layer(x, l)

	return x

shared_variables = init_variables(shared_shape)
shared_layers = lambda x: nn(shared_variables, x, True, False)

class PrivateDomain:
	def __init__(self,
		data,
		ind=0,
		tagged=False,
		classes=[0,1],
		weight=1,
		delay=1):

		self.data = data
		self.weight = weight
		self.tagged = tagged
		self.delay = delay
		self.classes = classes
		self.encoder_shape = [self.data.count(), self.data.count(), self.data.count(), num_hidden_1, num_hidden_1, num_hidden_1]

		self.decoder_shape = shared_shape[::-1] + self.encoder_shape[::-1] 

		self.x = tf.placeholder(tf.float32, 
			shape=[batch_size, data.count()])

		self.feedable = [self.x]

		if tagged:
			self.labels = tf.placeholder(tf.float32,
				shape=[batch_size, 1])

			self.feedable.append(self.labels)

		self.init_vars()
	
	def init_vars(self):
		self.encoder_v = init_variables(self.encoder_shape)
		self.decoder_v = init_variables(self.decoder_shape)

	def run(self, x):
		encoded = nn(self.encoder_v, x, is_enc=True)
		squeezed = shared_layers(encoded)
		decoded = nn(self.decoder_v, squeezed)

		self.result = [x, encoded, squeezed, decoded]

		return recon_loss(x, decoded)

	def loss(self,
		global_step):

		batch_loss = self.run(self.x)

		estimate_cond = global_step % tf.to_int32(self.delay)
		self.dec_loss = tf.cond(tf.equal(estimate_cond, 0),
			lambda: batch_loss,
			lambda: tf.to_float(0.0))

		if self.tagged:
			self.class_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.labels,
					logits=self.logits)
		else:
			self.class_loss = 0.0

		return tf.reduce_sum([self.weight * self.dec_loss, self.class_loss])

	def feed_dict(self, step):
		if not step % self.delay:
			vals = next(self.data)
		else:
			vals = self.data.placeholders()

		feed_dict = {k: v for k, v in zip(self.feedable, vals)}

		return feed_dict

	def feed_valid_dict(self):
		feed_dict = {k: v for k, v in zip(self.feedable, self.data.validation_set())}

		return feed_dict