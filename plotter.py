import matplotlib.pyplot as mp
import numpy as np
from sklearn.manifold import TSNE

def plot(x, y, title, pos):
	sub_coord = 230 + pos
	mp.subplot(sub_coord)
	mse = ((x - y) ** 2).mean()
	mp.scatter(x, y, label=str(mse))
	mp.xlabel('Original')
	mp.ylabel('Decoded')
	mp.title(title)
	mp.legend(loc=2)

paths = ['model0.0_res_', 'model1.0_res_']
suff = ['train.npz', 'valid.npz']


with np.load('model0.0_res_train.npz') as f:
	decoded = f['arr_0']
	x00 = f['arr_1']
	print(len(x00))
	plot(x00, decoded, "Main dataset", 1)

with np.load('model1.0_res_train.npz') as f:
	decoded0 = f['arr_0']
	x0 = f['arr_1']
	plot(x0, decoded0, "Small dataset", 2)

with np.load('model0.0_res_valid.npz') as f:
	decoded = f['arr_0']
	x11 = f['arr_1']
	print(len(x11))
	plot(x11, decoded, "Main dataset", 3)

with np.load('model1.0_res_valid.npz') as f:
	decoded1 = f['arr_0']
	x1 = f['arr_1']
	plot(x1, decoded1, "Small dataset", 4)

with open("80955_filtered.txt") as f:
	x0 = f.readline().split()
	x0 = [float(x) for x in x0]
	x1 = f.readline().split()
	x1 = [float(x) for x in x1]

plot(np.asarray(x0), np.asarray(x1), "Small dataset scatter", 5)

mp.show()

