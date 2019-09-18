import dask.dataframe as dd
import pandas as pd
import numpy as np
import tensorflow as tf

class Data: 

	def __init__(self,
		filename,
		sep=',',
		ind=0,
		split=False,
		split_start=None,
		split_end=None,
		batch_size=50,
		log=False,
		additional_info=None,
		left_on='sample',
		right_on='Sample'):

		"""Create a new dataset.

		Parametrs:
			filename -- csv file containting gene expression data
			sep -- separator used in the filename (default ',')
			split -- whether the dataset is splitted into two datasets with different
			 columns (genes)
			split_start -- index of the first column of the split
			split_end -- index of the last column of the split
			batch_size -- batch size
			log -- True if the data should be log transformed
			left_on -- how the column with sample names is called in dataset
			right_on -- how the column with sample names is called in file with samples descriptions
		"""


		self.df = pd.read_csv(filename, sep=sep)
		self.ind = ind
		self.tagged = True if additional_info else False

		if log:
			self.df.iloc[:, 2:] = self.df.iloc[:, 2:].applymap(lambda x: np.log(x + 1))
		
		if additional_info:
			tags_df = pd.read_csv(additional_info,
				sep=' ')

			#Add the tags to the main dataset
			self.df = pd.merge(self.df, tags_df, 
				left_on=left_on, 
				right_on=right_on,
				how='outer').iloc[:,:-1]
			self.num_classes = self.df['Disease'].nunique() - 1

			#Convert categorical variables to indicators
			self.df = pd.concat([self.df.drop('Disease', axis=1), 
				pd.get_dummies(self.df['Disease'],
					drop_first=True)], axis=1)

		if split:
			splitted_df = self.df.iloc[:, split_start:split_end]

			#If there is no column with sample names in this split then add one
			if 'sample' not in splitted_df.columns:
				splitted_df.insert(0, 'sample', self.df['sample'])

			self.df = splitted_df
			
		self.batch_size = batch_size
		self.dim = len(self.df.columns) - 2
		self.placeholder = tf.placeholder(tf.float32, 
			shape=[batch_size, self.dim])
	
		self._create_valid()


	def _create_valid(self):
		""" Create valid batch"""

		self.valid_indices = np.random.random_integers(0, 
			len(self.df) - 1, 
			size=(self.batch_size, ))
		
		self.valid = self.df.iloc[self.valid_indices, 2:-self.num_classes if self.tagged else None]

		if self.tagged:
			self.valid_tags = self.df.iloc[self.valid_indices, -self.num_classes:]

		# Drop the validation data from the training dataset.
		self.original_df = self.df
		self.df = self.df.drop(self.valid_indices)

	def second_split(self, second_frame):
		""" Get validation data from the second part of the dataframe
		Parametrs:
			second_frame -- second part of the dataframe
		"""
		return second_frame.iloc[self.valid_indices, 2:].values 

	def __next__(self):
		""" Return the next batch of training data """

		if self.tagged:
			data = self.df.sample(n=self.batch_size).iloc[:, 2:]
			tags = data.iloc[:,-self.num_classes:].values
			values = data.iloc[:, :-self.num_classes]

			return values, tags

		return [self.df.sample(n=self.batch_size).iloc[:, 2:].values]


	def placeholders(self):
		""" Fake data """
		
		batch = [[0 for _ in range(self.dim)]] * self.batch_size

		if self.tagged:
			labels = [0 for _ in range(self.num_classes) - 1] * self.batch_size

			return batch, labels

		return batch

