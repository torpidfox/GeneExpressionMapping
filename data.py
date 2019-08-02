import dask.dataframe as dd
import pandas as pd
import numpy as np
import tensorflow as tf

class Data: 

	def __init__(self,
		filename,
		sep=',',
		batch_size=50,
		padding_size=1,
		tags=None,
		log=False,
		norm=False,
		exp=False,
		additional_info=None,
		left_on=None,
		right_on=None):


		self.df = pd.read_csv(filename, sep=sep)
		self.dim = len(self.df.columns) - 2
		self.placeholder = tf.placeholder(tf.float32, 
			shape=[batch_size, self.dim])

		self.tagged = True if additional_info else False

		if log:
			self.df.iloc[:, 2:] = self.df.iloc[:, 2:].applymap(lambda x: np.log(x + 1))
		
		if additional_info:
			tags_df = pd.read_csv(additional_info,
				sep=' ')
			self.df = pd.merge(self.df, tags_df, 
				left_on='sample', 
				right_on='Sample',
				how='outer').iloc[:,:-1]
			self.num_classes = self.df['Disease'].nunique() - 1

			self.df = pd.concat([self.df.drop('Disease', axis=1), 
				pd.get_dummies(self.df['Disease'],
					drop_first=True)], axis=1)

		self.batch_size = batch_size
		self.padding_size = padding_size
		self.classes = tags if tags else False
		
		self._create_valid()


	def _create_valid(self):
		self.valid_indices = np.random.random_integers(0, 
			len(self.df) - 1, 
			size=(self.batch_size, ))
		
		self.valid = self.df.iloc[self.valid_indices, 2:-self.num_classes if self.tagged else None]

		if self.tagged:
			self.valid_tags = self.df.iloc[self.valid_indices, -self.num_classes:]

		self.df = self.df.drop(self.valid_indices)

	def __next__(self):
		if self.tagged:
			data = self.df.sample(n=self.batch_size).iloc[:, 2:]
			tags = data.iloc[:,-self.num_classes:].values
			values = data.iloc[:, :-self.num_classes]

			return values, tags

		return [self.df.sample(n=self.batch_size).iloc[:, 2:].values]


	def placeholders(self):
		batch = [[0 for _ in range(self.dim)]] * self.batch_size

		if self.tagged:
			labels = [0 for _ in range(self.classes) - 1] * self.batch_size

			return batch, labels

		return batch

