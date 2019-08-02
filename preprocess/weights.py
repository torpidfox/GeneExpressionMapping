import numpy as np
import json

l = 0.5

buff = []
with open('../test_data/pathways.txt') as f:
	f.readline()

	for line in f:
		buff += line.split()
		f.readline()

genes = set(buff)


gene_to_gene = {gene : {g : 0 for g in genes} for gene in genes}

with open('../test_data/pathways.txt') as f:
	f.readline()

	for line in f:
		path = line.split()

		for g1 in path:
			for i, g2 in enumerate(path):
				if g1 != g2:
					gene_to_gene[g1][g2] += np.exp(-l * i)


with open('../test_data/weights.txt', 'w') as f:
	json.dump(gene_to_gene, f)
