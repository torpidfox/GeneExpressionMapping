from subset import PrivateDomain
import tensorflow as tf
from logger import Logger

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

		delay_steps = [self.epoch_step % tf.to_int32(s.delay) for s in self.sets]
		distr_loss = list()

		for s1, delay_step1 in zip(self.sets, delay_steps):
			_, var1 = tf.nn.moments(s1.result[0], axes=[0])

			for s2, delay_step2 in zip(self.sets, delay_steps):
				if s1 != s2:
					should_add = tf.logical_and(tf.equal(delay_step1, 0),
						tf.equal(delay_step2, 0))

					_, var2 = tf.nn.moments(s2.result[0], axes=[0])
					coeff = tf.sqrt(tf.reduce_mean(var1) * tf.reduce_mean(var2))

					distr_loss.append(tf.cond(should_add,
						lambda: tf.losses.mean_squared_error(s2.result[1], s1.result[1]),
						lambda: tf.to_float(0.0)
						))

		self.loss = distr_loss+private_loss

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
		opt = tf.train.AdagradOptimizer(learning_rate=1e-3).minimize(tf.reduce_sum(self.losses()))
		# gvs = optimizer.compute_gradients(tf.reduce_sum(self.losses()))		
		# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
		# opt = optimizer.apply_gradients(gvs)

		valid_op = [self.loss] + [s.result for s in self.sets]
		train_op = valid_op + [opt]

		return train_op, valid_op, step_inc_op

	def run(self):
		train_op, valid_op, step_inc_op = self.build_graph()
		valid_loss_step = 0
		min_valid_loss = float('inf')
		valid_loss = 10
		step = 0

		with Logger('red_new') as logging:
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())	
				sess.run(tf.local_variables_initializer())

				for var in tf.trainable_variables():
					print(var.name)	

				while valid_loss_step < 100:
					#variables_names = [v.name for v in tf.trainable_variables()]
					#values = sess.run(variables_names)

					step = sess.run(step_inc_op)
					feed_dict = self.feed_dict(step)	
					
			
					#for j in range(iter_count):				
					loss, *result, _ = sess.run(train_op, feed_dict=feed_dict)

					if not step % 10:
						print(step, loss)
						print("Epoch: {}, distr loss: {}, recon loss 1: {}, recon loss 2: {}".format(step, loss[0], loss[2], loss[3]))
						#print('valid', valid_loss, valid_loss_step)
						valid_loss, *valid_result = sess.run(valid_op, feed_dict=self.valid)
						if sum(valid_loss) < min_valid_loss:
							min_valid_loss = sum(valid_loss) 
							valid_loss_step = 0
							logging.dump_res(valid_result, attr='valid')
						else:
							valid_loss_step += 1

					logging.log_results(step, [loss, valid_loss])
						
						

def sess_runner(sets):
	m = Model(sets)

	m.run()






		






