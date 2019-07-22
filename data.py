import pandas as pd
import numpy as np

class Data: 

	def __init__(self,
		filename,
		sep=',',
		batch_size=100,
		padding_size=1,
		tags=None,
		log=False,
		norm=False,
		exp=False,
		additional_info=None):


		self.df = pd.read_csv(filename, sep=sep)
		self.dim = len(self.df.columns) - 2
		self.placeholder = tf.placeholder(tf.float32, 
			shape=[None, self.dim])

		if log:
			self.df.iloc[:, 2:] = self.df.iloc[:, 2:].applymap(np.log)
			self.df.replace(-np.inf, 0, inplace=True)


		self.batch_size = batch_size
		self.padding_size = padding_size
		
		self.info = dd.read_csv(additional_info) if additional_info else None
		self.classes = tags if tags else False
		
		self._create_valid()


	def _create_valid(self):
		self.valid_indices = np.random.random_integers(0, 
			len(self.df), 
			size=(self.batch_size, ))

		self.valid = self.df.iloc[self.valid_indices, 2:]
		self.df = self.df.drop(self.valid_indices)

	def __next__(self):
		return self.df.sample(n=self.batch_size).iloc[:, 2:].values


	def placeholders(self):
		batch = [[0 for _ in range(self.dim)]] * self.batch_size

		if self.tagged:
			labels = [0 for _ in range(self.classes) - 1] * self.batch_size
		else:
			labels = None

		return batch, labels

