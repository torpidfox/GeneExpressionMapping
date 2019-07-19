import dask.dataframe as dd
import pandas as pd


class Data: 

	def __init__(self,
		filename,
		sep='\t',
		batch_size=100,
		padding_size=1,
		reuse=False,
		tags=None,
		valid=True,
		log=False,
		norm=False,
		exp=False,
		additional_info=None):

		df = dd.read_csv(filename, sep=sep)
		self.batch_size = batch_size
		self.padding_size = padding_size
		self.valid = None
		self.dim = 0
		self.reuse = reuse
		self.info = dd.read_csv(additional_info) if additional_info else None
		self.classes = tags if tags else False
		self.create_valid = valid
		self.log = log
		self.norm = norm
		self.exp = exp

		print(df.head(3))


Data(filename='test_data/GSE80655_GeneExpressionData_Updated_3-26-2018.txt')




