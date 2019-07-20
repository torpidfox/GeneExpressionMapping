import pandas as pd

genes = []
with open('../test_data/pathways.txt') as f:
	f.readline()

	for l in f:
		genes += l.split()
		f.readline()

kegg_genes = set(genes)

name_mapped = pd.read_csv('../test_data/affy_to_kegg', sep=' ')
print(dict(zip(name_mapped.probe_id, name_mapped.symbol)))
print(name_mapped.to_dict())

genes = []

# with open('../data/processedMatrix.Aurora.july2015.txt') as input_f, open('../data/kegg_filtered.txt', 'w') as output_f:
# 	samples = input_f.readline()
# 	output_f.write(samples)

# 	for l in input_f:
# 		gene_data = l.split()
# 		print(name_mapped.loc[df['probe_id'] == gene_data[0]]['symbol'])

# 		if name_mapped.loc[df['probe_id'] == gene_data[0]]['symbol'] in kegg_genes:
# 			print(gene_data[0])
# 			output_f.write(gene_data.join(' '))

