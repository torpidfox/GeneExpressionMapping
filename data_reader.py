import numpy as np

class Data:
	""" class preparing feeding data for model """

	def __init__(self, filenames=[], batch_size=150, padding_size=1):
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
		self._load_next()

	def _load_next(self):
		"""load next file"""

		with open(self.filenames.pop(), 'r') as f:
			lines = [l.split() for l in f.readlines()]
			lines = [list(map(float, el)) for el in lines]
			print(len(lines))
			self.dim = len(lines[0])

		self.file = [lines[i:i + self.batch_size] * self.padding_size for i in range(0, len(lines), self.batch_size)]
		self.file.pop(-1)
		print(len(self.filenames))
		print(len(self.file))
		print(len(self.file[0]))

	def count(self):
		"""get sample size"""
		
		return self.dim

	def __next__(self):
		"""get next batch"""
		if len(self.file) == 1:
			self._load_next()

		return np.asarray(self.file.pop())
