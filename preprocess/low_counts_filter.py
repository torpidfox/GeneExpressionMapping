import pandas as pd

df = pd.read_csv('../test_data/80655_norm.txt', sep=' ', header=0)

genes = []
with open('../test_data/pathways.txt') as f:
	f.readline()

	for l in f:
		genes += l.split()
		f.readline()

print(len(set(genes)))
header = list(df)
sums = df.sum(axis=0)
intercept = [el for el in set(genes) if el in header and sums[el] > 10]
filtered_df = df[['sample'] + intercept]
print(len(list(filtered_df)))
filtered_df.to_csv('../test_data/GSE80655_filtered.csv')
