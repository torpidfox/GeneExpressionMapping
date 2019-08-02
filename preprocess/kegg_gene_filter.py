import pandas as pd

genes = []
with open('../test_data/pathways.txt') as f:
	f.readline()

	for l in f:
		genes += l.split()
		f.readline()

kegg_genes = set(genes)

name_mapped = pd.read_csv('../test_data/affy_to_kegg_2', sep=' ')
name_mapped = dict(zip(name_mapped.probe_id, name_mapped.symbol))

genes = []

with open('../test_data/45642_conc.csv') as input_f, open('../test_data/45642_filtered.txt', 'w') as output_f:
	samples = input_f.readline()
	output_f.write(samples)

	for l in input_f:
		gene_data = l.split(',')
		print(name_mapped[gene_data[1]])

		if name_mapped[gene_data[1]] in kegg_genes:
			print(gene_data[1])
			output_f.write(' '.join(gene_data) + '\n')

