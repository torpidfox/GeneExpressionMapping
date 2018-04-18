import numpy as np

class Data:
	""" class preparing feeding data for model """

	def __init__(filenames=[], batch_size=200, padding_size=1):
		"""
		keywords args:
		filenames -- source files
		batch_size -- actual batch size for dataset
		padding_size -- how many batch_size pieces are required to pad batch for model
		"""

		self.filenames = filenames
		self.batch_size = batch_size
		self.padding_size = padding_size
		self.file = None
		self.dim = 0

	def _load_next(self):
		"""load next file"""

		with open(self.filenames.pop(), 'r') as f:
			lines = np.asarray([l.split() for l in f.readlines()])
			self.dim = len(lines[0])
			file = [np.asfloat(line) for line in lines]

		batch_count = len(file) % self.batch_size
		self.file = iter(np.reshape(file, (self.batch_size, batch_count)))

	def count(self):
		"""get sample size"""
		return self.dim

	def __next__(self):
		"""get next batch"""

		if self.pos == len(self.file):
			_load_next()

		return np.flatten(self.next() * padding_size)
