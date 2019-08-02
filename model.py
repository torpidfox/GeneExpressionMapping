from subset import PrivateDomain
import tensorflow as tf
from logger import Logger

import sys
sys.path.append('./add_libs/')
import losses as los
summaries_dir = './summary/'

tres_valid_loss = 20
iter_count = 50

class Model:
	def __init__(self, sets):
		self.sets = sets

		self.valid = dict()

		for s in self.sets:
			self.valid.update(s.feed_valid_dict())

	def losses(self):
		private_loss = [s.loss(self.epoch_step) for s in self.sets]
		class_loss = [s.class_loss for s in self.sets if s.tagged]
		accuracy = [s.accuracy for s in self.sets if s.tagged]

		delay_steps = [self.epoch_step % tf.to_int32(s.delay) for s in self.sets]
		distr_loss = list()

		for s1, delay_step1 in zip(self.sets, delay_steps):
			_, var1 = tf.nn.moments(s1.result[0], axes=[0])

			for s2, delay_step2 in zip(self.sets, delay_steps):
				if s1 != s2:
					should_add = tf.logical_and(tf.equal(delay_step1, 0),
						tf.equal(delay_step2, 0))

					_, var2 = tf.nn.moments(s2.result[0], axes=[0])
					coeff = 1.0 / tf.sqrt(tf.reduce_mean(var1) * tf.reduce_mean(var2))

					distr_loss.append(tf.cond(should_add,
						lambda: tf.losses.mean_squared_error(s2.result[1], s1.result[1]) / 2,
						lambda: tf.to_float(0.0)
						))

		#self.loss = [tf.to_float(0.0), tf.to_float(0.0)] + private_loss
		# reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		# reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
		self.loss = distr_loss+private_loss+class_loss+accuracy
		print(self.loss, accuracy, class_loss, private_loss)
		tf.summary.scalar('distribution loss', distr_loss)
		
		return self.loss

	def feed_dict(self, step):
		feed_vals = dict()

		for s in self.sets:
			feed_vals.update(s.feed_dict(step))

		return feed_vals



	def build_graph(self):
		print('b')
		for s in self.sets:
			s.run(s.x)

		self.epoch_step = tf.Variable(0,
			name='epoch_num',
			trainable=False)

		step_inc_op = tf.assign_add(self.epoch_step, 1)

		learning_rate = tf.train.exponential_decay(1e-2, self.epoch_step, 100, 0.96)

		with tf.name_scope('loss'):
			total_loss = tf.reduce_sum(self.losses())
		tf.summary.scalar('loss', total_loss)

		opt = tf.train.AdagradOptimizer(learning_rate=1e-3).minimize(total_loss)

		valid_op = [self.loss] + [s.result for s in self.sets]
		train_op = valid_op + [opt]

		return train_op, valid_op, step_inc_op

	def run(self):
		train_op, valid_op, step_inc_op = self.build_graph()
		valid_loss_step = 0
		min_valid_loss = float('inf')
		valid_loss = 10
		step = 0
		saver = tf.train.Saver()

		with Logger('75140') as logging:
			with tf.Session() as sess:
				#summary_op = tf.merge_all_summaries()
				#train_writer = tf.train.SummaryWriter(summaries_dir + '/train', sess.graph)
				#test_writer = tf.train.SummaryWriter(summaries_dir + '/test')

				sess.run(tf.global_variables_initializer())	
				sess.run(tf.local_variables_initializer())

				while valid_loss_step < 300:
					step = sess.run(step_inc_op)
					feed_dict = self.feed_dict(step)

					loss, *result, _ = sess.run(train_op, feed_dict=feed_dict)
					#train_writer.add_summary(summary, step)


					if not step % 10:
						print(step, loss)
						# print("Epoch: {}, distr loss: {}, recon loss 1: {}, recon loss 2: {}, regular loss: {}"
						# 	.format(step, loss[0], loss[2], loss[3], loss[4]))
						
						print('valid', valid_loss, valid_loss_step)
						valid_loss, *valid_result = sess.run(valid_op, feed_dict=self.valid)
						#test_writer.add_summary(summary, step)

						if sum(valid_loss) < min_valid_loss:
							min_valid_loss = sum(valid_loss) 
							valid_loss_step = 0
							logging.dump_res(valid_result, attr='valid')
						else:
							valid_loss_step += 1

					if not step % 1000:
						saver.save(sess, "./tmp/model.ckpt")

					logging.log_results(step, [loss, valid_loss])
						
						

def sess_runner(sets):
	m = Model(sets)

	m.run()






		






