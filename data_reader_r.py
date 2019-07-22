import numpy as np
import pandas as pd
import random
from math import exp

class Data:
	""" class preparing feeding data for model """

	def __init__(self,
		filenames=[],
		batch_size=100,
		padding_size=1,
		reuse=False,
		tags=None,
		valid=True,
		log=False,
		norm=False,
		exp=False,
		additional_info=None):
		
		"""
		keywords args:
		filenames -- source files
		batch_size -- actual batch size for dataset
		padding_size -- how many batch_size pieces are required to pad batch for model
		reuse -- whether to reuse previous batches
		additional_info -- filename for data descriptions
		"""

		self.filenames = filenames
		self.batch_size = batch_size
		self.padding_size = padding_size
		self.file = None
		self.valid = None
		self.dim = 0
		self.reuse = reuse
		self.informed = False if not additional_info else True
		self.info = None
		self.cur_file = 0
		self.tagged = True if tags else False
		self.classes = tags
		self.create_valid = valid
		self.log = log
		self.norm = norm
		self.exp = exp
	

		if additional_info:
			with open(additional_info) as f:
				self.info = pd.read_csv(f, sep='	')
			
			self.cur_ind = 0

		self._load_next(self.classes)

	def _normalize_sample(self, d):
		# d is a (n x dimension) np array
		d -= np.min(d, axis=0)
		d /= np.ptp(d, axis=0)

		return d

	def normalize(self, data):
		res = self._normalize_sample(data)

		return res
		


	def _load_next(self, tags=None):
		"""load next file"""

		cur_file = self.filenames[self.cur_file] if not self.reuse else self.filenames[0]

		with open(cur_file, 'r') as f:
			lines = f.readlines()
			lines = list(map(lambda x: x.split(), lines))
			names = [el[0] for el in lines]
			if self.log:
				lines = [list(map(lambda x: np.log(float(x) + 1), el[1:])) for el in lines]
			elif self.exp:
				lines = [list(map(lambda x: np.exp(float(x)) - 1, el[1:])) for el in lines]

			else:
				lines = [list(map(lambda x: float(x), el[1:])) for el in lines]

			lines = np.asarray(lines)

			lines = lines[:, ~np.all(lines < 1e-3, axis=0)]

			if self.norm:
				lines = self.normalize(lines)
			self.file = list(zip(names, lines))

			self.dim = len(lines[0])
		if self.tagged:
			data = self.info[self.info['Disease'].isin(tags)]
			data = data['Ind'].tolist()
			self.file = list(filter(lambda x: x[0] in data, self.file))
		if self.create_valid and not self.valid:
			self.valid = self.file[-self.batch_size:] * self.padding_size
			self.file = self.file[:-self.batch_size]
		else:
			self.valid = list(zip([[0]*1]*self.batch_size*self.padding_size, [[0]*self.dim]*self.batch_size*self.padding_size))
			self.valid_tags = [[0]*1]*self.batch_size*self.padding_size

		self.cur_file = self.cur_file if not self.reuse else self.cur_file + 1
		if self.cur_file == len(self.filenames):
			self.cur_file = 1

	def reset(self):
		self.cur_file = 0


	def _tags(self, column, classes, arr, padding=1):
		tags = self.info[self.info['Ind'].isin([s[0] for s in arr])]
		res = list()
		for _, el in tags.iterrows():
			el_tag = list()
			for c in classes[1:]:
				if el[column] == c:
					el_tag.append(1)
				else:
					el_tag.append(0)

			while len(el_tag) < padding:
				el_tag.append(0)

			res.append(el_tag)

		return np.asarray(res * self.padding_size)

	def tags(self, column, classes, padding=1):
		return self._tags(column, classes, self.file[:self.batch_size], padding)

	def v_tags(self, column, classes, padding=1):
		if not self.create_valid:
			return self.valid_tags

		return self._tags(column, classes, self.valid, padding)


	def count(self):
		"""get sample size"""
		
		return self.dim

	def validation_set(self): 
		"""get validation set"""
		if self.tagged:
			return [sample[1] for sample in self.valid], self.v_tags()

		return [[sample[1] for sample in self.valid]]

	def tagged_samples(self, column, tag, count):
		""" load samples with certain tag """

		data = self.info[self.info['Disease'].isin(tag)]
		data = data['Ind'].tolist()

		i = 0
		res = list()
		ind = 0
		file_ind = 0

		while file_ind < len(self.filenames):
			with open(self.filenames[file_ind]) as file:
				for f in file:
					line = f.replace(',', '').split()
					if line[0] in data:
						line = f.replace(',', '').split()
						line = list(map(float, line[1:]))
						res.append(line)

					ind += 1

				file_ind += 1

		while len(res) < count:
			res += res

		return np.asarray(res[:count])


	def __next__(self):
		"""get next batch"""
		if not self.reuse:
			del(self.file[:self.batch_size])
			if len(self.file) == 0:
				self._load_next(self.classes)

		random.shuffle(self.file)
		batch = [s[1] for s in self.file[:self.batch_size]] * self.padding_size

		if self.tagged:
			return np.asarray(batch), self.tags()

		return [np.asarray(batch)]


	def create_tagged_dataset(self, tags):
		inds = list()

		for t in tags:
			inds+=list(self.info.index[self.info["Characteristics[organism part]"] == t])

		i = 0
		data = list()
		ind = 0
		file_ind = 0

		for f in self.filenames:
			print(f)
			with open(f) as file:
				for l in file:
					if ind in inds:
						line = l.split()
						line = list(map(lambda x: x.replace(',', '')
							.replace('"', '')
							.replace('[','')
							.replace(']', '')
							.replace('\'', '')
							.split(), lines))
						if not self.log:
							data.append(line)
						else:
							data.append(list(map(lambda x: np.log(x + 1), line)))

					ind += 1
		iter_range = range(0, len(data) - self.batch_size, self.batch_size) 
		self.valid = data[-self.batch_size:]
		self.file = [data[i:i+self.batch_size]*self.padding_size for i in iter_range] 
		self.dim = len(self.valid[0])

	def placeholders(self):
		batch = [[0 for _ in range(self.dim)]] * self.batch_size

		if self.tagged:
			labels = [0 for _ in range(self.classes) - 1] * self.batch_size
		else:
			labels = None

		return batch, labels

