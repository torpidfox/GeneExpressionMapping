from subset import PrivateDomain, nn
import tensorflow as tf
from logger import Logger

import sys
sys.path.append('./add_libs/')
import losses as los
summaries_dir = './summary/'
logs_dir = 'two_sets_from_one'

tres_valid_loss = 1
iter_count = 1
dropout_prob = tf.get_default_graph().get_tensor_by_name('dropout_prob:0')

squeezed_data_ind = -2


class Model:
	def __init__(self, sets, split=False):
		self.sets = sets
		self.valid = {dropout_prob : 1}
		self.split = split

		for s in self.sets:
			self.valid.update(s.feed_valid_dict())

	def losses(self):
		""" Compute all the losses """

		# private decoders loss
		private_loss = [s.loss(self.epoch_step) for s in self.sets]

		# private classification loss
		class_loss = [s.class_loss for s in self.sets if s.tagged]

		# private accuracy
		self.accuracy = [s.accuracy for s in self.sets if s.tagged]

		delay_steps = [self.epoch_step % tf.to_int32(s.delay) for s in self.sets]
		distr_loss = list()

		# append the result of applying one split's decoder to another split and vice versa
		if self.split:
			self.sets[0].result.append(
				nn(self.sets[0].decoder_v, self.sets[1].result[squeezed_data_ind])
			)
			self.sets[1].result.append(
				nn(self.sets[1].decoder_v, self.sets[0].result[squeezed_data_ind - 1])
			)

		# compute distribution losses
		if not self.split:
			for s1, delay_step1 in zip(self.sets, delay_steps):
				_, var1 = tf.nn.moments(s1.result[0], axes=[0])

				for s2, delay_step2 in zip(self.sets, delay_steps):
					if s1 != s2:	
						should_add = tf.logical_and(tf.equal(delay_step1, 0),
							tf.equal(delay_step2, 0))

						_, var2 = tf.nn.moments(s2.result[0], axes=[0])
						coeff = 1.0 / tf.sqrt(tf.reduce_mean(var1) * tf.reduce_mean(var2))

						distr_loss.append(tf.cond(should_add,
							lambda: tf.to_float(0.0),	
							lambda: tf.losses.mean_squared_error(s2.result[1], s1.result[1]) / 2,
							))

		#self.loss = [tf.to_float(0.0), tf.to_float(0.0)] + private_loss

		#total loss
		self.loss = distr_loss+private_loss+class_loss
		tf.summary.scalar('distribution loss', distr_loss)
		
		return self.loss

	def feed_dict(self, step):
		""" returns the dictionary to feed to tensorflow's session to 
		Parametrs:
			step -- global session step

		""" 
		feed_vals = {dropout_prob : 0.8}

		# add training batch from all the sets
		for s in self.sets:
			feed_vals.update(s.feed_dict(step))

		return feed_vals


	def build_graph(self):
		""" Perform all the computations """

		for s in self.sets:
			s.run(s.x)

		# global step variable
		self.epoch_step = tf.Variable(0,
			name='epoch_num',
			trainable=False)

		step_inc_op = tf.assign_add(self.epoch_step, 1)

		learning_rate = tf.train.exponential_decay(1e-2, self.epoch_step, 100, 0.96)

		with tf.name_scope('loss'):
			total_loss = tf.reduce_sum(self.losses())

		tf.summary.scalar('loss', total_loss)

		opt = tf.train.AdagradOptimizer(learning_rate=1e-3).minimize(total_loss)

		valid_op = [self.loss] + [self.accuracy] + [s.result for s in self.sets]
		train_op = valid_op + [opt]

		return train_op, valid_op, step_inc_op

	def run(self):
		""" Run training """

		train_op, valid_op, step_inc_op = self.build_graph()
		valid_loss_step = 0
		min_valid_loss = float('inf')
		valid_loss = 10
		valid_acc = 0
		step = 0
		saver = tf.train.Saver()

		with Logger(logs_dir) as logging:
			with tf.Session() as sess:
				#summary_op = tf.merge_all_summaries()
				#train_writer = tf.train.SummaryWriter(summaries_dir + '/train', sess.graph)
				#test_writer = tf.train.SummaryWriter(summaries_dir + '/test')

				sess.run(tf.global_variables_initializer())	
				sess.run(tf.local_variables_initializer())

				while valid_loss_step < tres_valid_loss:
					step = sess.run(step_inc_op)
					feed_dict = self.feed_dict(step)
					loss, acc, *result, _ = sess.run(train_op, feed_dict=feed_dict)

					if not step % 10:
						print(step, loss, acc)
						print('valid', valid_loss, valid_acc, valid_loss_step)
						valid_loss, valid_acc, *valid_result = sess.run(valid_op, feed_dict=self.valid)

						if sum(valid_loss) < min_valid_loss:
							min_valid_loss = sum(valid_loss) 
							valid_loss_step = 0

							# append the source data from the second split to compare to the results of decoding
							valid_result[0].append(self.sets[0].data.second_split(self.sets[1].data.original_df))	
							valid_result[1].append(self.sets[1].data.second_split(self.sets[0].data.original_df))	
							logging.dump_res(valid_result, attr='valid')
							
							if not step % 1000:
								saver.save(sess, "../logs/{}/model.ckpt".format(logs_dir))
						else:
							valid_loss_step += 1


					logging.log_results(step, [loss, valid_loss])
						
						

def sess_runner(sets):
	m = Model(sets, split=True)

	m.run()






		






