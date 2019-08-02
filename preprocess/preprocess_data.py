import tensorflow as tf
import pandas as pd
import numpy as np
import json

def filter_genes(genes, weights):
	return {g1 : {g2 : weights[g1][g2] for g2 in genes} for g1 in genes}

def create_weights(df, weights):
	w = []
	genes = list(df)[2:]
	weights_filtered = filter_genes(genes, weights)

	for g in genes:
		w.append(list(weights_filtered[g].values()))

	return np.array(w)

def preprocess(df, weights):
	result = df.iloc[:,2:]
	print(result)
	print(result.dot(weights.T))
	return df.dot(weights)

df = pd.read_csv('../test_data/GSE80655_filtered.csv')
with open('../test_data/weights.txt') as f:
	weights = json.load(f)

print(print(preprocess(df, create_weights(df, weights))))