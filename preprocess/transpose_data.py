import pandas as pd


prev_i = 1

with open('3732_transposed.txt', 'w') as f:
	pass

with open('3732_transposed.txt', 'a') as f:
	for i in range(100, 200, 100):
		df = pd.read_csv('../data/kegg_filtered.txt', 
			sep=' ', 
			skiprows=1, 
			header=0,
			names=range(1:27888),
			usecols=lambda x: x in [prev_i:i])

		transposed = df.T
		transposed.to_csv(f, header=True if prev_i == 1 else False)
		prev_i = i
	