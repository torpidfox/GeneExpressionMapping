from data_reader_r import Data
import tensorflow as tf
import numpy as np 

init = tf.contrib.layers.xavier_initializer()

shared_scope = 'shared'
gene_count = 1000
num_hidden_2 = gene_count // 4
num_hidden_1 = gene_count // 2
batch_size = 50
num_classes = 4 
dropout_prob = tf.placeholder_with_default(1.0, 
	shape=(),
	name='dropout_prob')

# neural network's params variables
shared_shape = [num_hidden_1, num_hidden_1, num_hidden_2,  num_hidden_2, num_hidden_2]
classification_shape = [num_hidden_2, num_classes]
activation = tf.nn.selu
init_weights = lambda n1, n2: tf.Variable(
            tf.random_normal([n1, n2], 0, np.sqrt(2 / n1))
            )

init_zeros = lambda n1: tf.Variable([0] * n1, dtype = 'float')
layer = lambda x, v: tf.nn.xw_plus_b(x, v['w'], v['b'])
recon_loss = lambda x1, x2: tf.losses.mean_squared_error(x1, x2)

def init_variables(shape):
	""" Init network's variables """
	variable_list = list()

	for i, dim in enumerate(shape[:-1]):
		l = {'w' : init_weights(dim, shape[i + 1]),
		'b' : init_zeros(shape[i + 1])}
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l['w'])
		variable_list.append(l)

	return variable_list

def nn(layers, x, is_enc=False, is_private=True):
	""" apply number of layers to the data
	Parameters:
		layers -- weights and biases of the layers
		x -- data to apply network to
		is_enc -- is this an encoder
		is_private -- is this a non-shared part
		"""

	for i, l in enumerate(layers):
		if i != len(layers) - 1:
			x = activation(layer(x, l))
			x = tf.nn.dropout(x, dropout_prob) if i == 0 and is_enc else x 
		elif is_enc and is_private:
			x = activation(layer(x, l))
		else:
			# do not apply activation to the very output
			x = layer(x, l)

	return x

def classify(x, labels):
	""" Classification 
	Parameters:
		x -- data
		labels -- computed lables (before the sofrmax)
	"""

	logits = classification_layers(x)
	class_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
			logits=logits)

	correct_pred = tf.equal(
		tf.argmax(tf.nn.softmax(logits), 1), 
		tf.argmax(labels, 1)
		)

	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	return class_loss, accuracy


shared_variables = init_variables(shared_shape)
shared_layers = lambda x: nn(shared_variables, x, True, False)

calssification_variables = init_variables(classification_shape)
classification_layers = lambda x: layer(tf.nn.dropout(x, dropout_prob),
 calssification_variables[0])

class PrivateDomain:
	def __init__(self,
		data,
		ind=0,
		tagged=False,
		weight=1,
		delay=1):

		""" Create variables that are private to the set
		Params:
			data -- Data object containing the dataset
			ind -- index of the set
			tagged -- is there classification data
			weight -- coefficient for the reconstruction loss
			delay -- how many epochs should be skipped before applying nn to this set
		"""

		self.data = data
		self.weight = weight
		self.tagged = tagged
		self.delay = delay
		self.ind = ind

		self.encoder_shape = [self.data.dim, self.data.dim, num_hidden_1, num_hidden_1]
		self.decoder_shape = shared_shape[::-1] + self.encoder_shape[::-1] 

		self.x = self.data.placeholder
		self.feedable = [self.x]

		if tagged:
			self.labels = tf.placeholder(tf.float32,
				shape=[batch_size, self.data.num_classes])

			self.feedable.append(self.labels)

		self.init_vars()
	
	def init_vars(self):
		""" Initialize network's variables """

		self.encoder_v = init_variables(self.encoder_shape)
		self.decoder_v = init_variables(self.decoder_shape)

	def run(self, x):
		""" Apply the network to the data """

		encoded = nn(self.encoder_v, x, is_enc=True)
		squeezed = shared_layers(encoded)

		if self.tagged:
			self.class_loss, self.accuracy = classify(squeezed, self.labels)

		decoded = nn(self.decoder_v, squeezed)

		self.result = [x, encoded, squeezed, decoded]

		if self.tagged:
			self.result.append(self.labels)

		return recon_loss(x, decoded)

	def loss(self,
		global_step):
		""" Compute reconstruction and (if applicable) classification losses 
		Params:
			global_step -- global epoch step
		"""

		batch_loss = self.run(self.x)

		estimate_cond = global_step % tf.to_int32(self.delay)
		self.dec_loss = tf.cond(tf.equal(estimate_cond, 0),
			lambda: batch_loss,
			lambda: tf.to_float(0.0))

		if not self.tagged:
			self.class_loss = 0.0

		return tf.reduce_sum([self.weight * self.dec_loss])

	def feed_dict(self, step):
		""" Construct the dict to feed to the network """

		if not step % self.delay:
			vals = next(self.data)
		else:
			vals = self.data.placeholders()

		feed_dict = {k: v for k, v in zip(self.feedable, vals)}

		return feed_dict

	def feed_valid_dict(self):
		""" Construct the validation dict to feed to the network """

		feed_dict = {self.x : self.data.valid}

		if self.tagged:
			feed_dict.update({self.labels : self.data.valid_tags})

		return feed_dict
