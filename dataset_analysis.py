import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

def import_data(inds):
	data = list()
	all_data = list()
	i = 0

	for filename in ["batch{}.txt".format(i) for i in range(19)]:
		with open(filename) as f:			
			for l in f:
				sample = l.split()
				all_data.append(list(map(float, sample)))

				if i in inds:
					data.append(list(map(float, sample)))

				i+=1

	return data, all_data

def import_from_nn(inds, tag):
	filtered01 = list()
	filtered02 = list()

	i = 0

	with np.load("model{}_res_shared.npz".format(tag)) as f:
		data01 = f['arr_0']
		data02 = f['arr_1']


		for d1, d2 in zip(data01, data02):
			if i in inds:
				filtered01.append(d1)
				filtered02.append(d2)

	return np.asarray(filtered01), np.asarray(filtered02)


with open("E-MTAB-3732.sdrf.txt") as f:
	df = pd.read_csv(f, sep='	')

#keys = ["lung", "breast", "colon", "brain", 'blood']
keys = ['brain']

# data1_in = dict()
# data1_out = dict()
# for key in keys:
# 	data1_in[key], data1_out[key] = import_from_nn(list(df.index[df["Characteristics[organism part]"] == key]), 0)

#data1_in['blood'] = list()

# for key in blood_keys:
# 	temp1, temp2 = import_from_nn(list(df.index[df["Characteristics[organism part]"] == key]), 0)

# 	for el in temp1:
# 		data1_in['blood'].append(el)

# 	for el in temp2:
# 		data1_out['blood'].append(el)

data1_in, data1_out = import_from_nn(list(range(101)), 0)
data2_in, data2_out = import_from_nn(list(range(101)), 1)
print(((data1_in - data2_in) ** 2).mean() / np.mean(np.sqrt(np.var(data1_in, axis=0)) * np.mean(np.var(data2_in, axis=0))))
print(((data1_out - data2_out) ** 2).mean() / np.mean(np.sqrt(np.var(data1_out, axis=0)) * np.mean(np.var(data2_out, axis=0))))
# blood_in = list()
# blood_out = list()

# with np.load("model0_res_blood.npz") as f:
# 	data01 = f['arr_0']
# 	data02 = f['arr_1']
# 	data2_in+=f['arr_2']
# 	data2_out+=f['arr_3']

# 	for d1, d2 in zip(data01, data02):
# 		blood_in.append(d1)
# 		blood_out.append(d2)

fig = plt.figure() 
t = PCA(n_components=3)

transformed1_in = t.fit_transform(data1_in)
transformed1_out = t.fit_transform(data1_out)

transformed2_in = t.fit_transform(data2_in)
transformed2_out = t.fit_transform(data2_out)

# transformed_blood_in = t.fit_transform(blood_in)
# transformed_blood_out = t.fit_transform(blood_out)

ax = fig.add_subplot(121, projection='3d')

# for el, tag in zip(transformed1_in, keys):
# 	ax.scatter([p[0] for p in el], [p[1] for p in el], [p[2] for p in el], label=tag)

ax.scatter([p[0] for p in transformed1_in], [p[1] for p in transformed1_in], [p[2] for p in transformed1_in],
	label='brain')




# ax.scatter([p[0] for p in transformed_blood_in], [p[1] for p in transformed_blood_in], [p[2] for p in transformed_blood_in],
# 	label='blood')

ax.scatter([p[0] for p in transformed2_in], [p[1] for p in transformed2_in], [p[2] for p in transformed2_in],
	label='additional dataset')

ax.legend()
plt.title('Autoencoder in, error')

ax = fig.add_subplot(122, projection='3d')

ax.scatter([p[0] for p in transformed1_out], [p[1] for p in transformed1_out], [p[2] for p in transformed1_out],
	label='brain')

# ax.scatter([p[0] for p in transformed_blood_out], [p[1] for p in transformed_blood_out], [p[2] for p in transformed_blood_out],
# 	label='blood')


# for el, tag in zip(transformed1_out, keys):
# 	ax.scatter([p[0] for p in el], [p[1] for p in el], [p[2] for p in el], label=tag)

ax.scatter([p[0] for p in transformed2_out], [p[1] for p in transformed2_out], [p[2] for p in transformed2_out],
	label='additional dataset')
ax.legend()



plt.title('Autoencoder out')

#transformed.append(t.fit_transform(all_data))



plt.show()