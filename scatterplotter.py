import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

fig = plt.gcf()


def plot(x, y, title, pos, xl='Original', yl='Decoded'): 
	sub_coord = 220 + pos 
	ax = fig.add_subplot(sub_coord) 
	mse = ((x - y) ** 2).mean()
	ax.scatter(x, y, label=str(mse))
	plt.xlabel(xl) 
	plt.ylabel(yl) 
	plt.title(title) 
	plt.legend(loc=2)
	# plt.xlim(left=0, right=14)
	# plt.ylim(bottom=0, top=14)
	plt.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".1") 

with np.load('model0_res_valid.npz') as f: 
	d1 = f['arr_0'] 
	print(np.shape(d1))
	d2_1 = f['arr_3'] 
	means = [((x - y) ** 2).mean() for x, y in zip(d1, d2_1)]
	print(means)
	print(np.asarray(means).mean())
	plot(d1[2], d2_1[2], "Main dataset valid result", 3)


with np.load('model0_res_valid.npz') as f:
	d1 = f['arr_0'] 
	print(np.shape(d1))
	plot(d1[1], d1[15], "Main dataset random samples", 1, xl='Sample 1', yl='Sample 2')  

with np.load('model1_res_valid.npz') as f: 
	d1 = f['arr_0'] 

	print(np.shape(d1)) 
	plot(d1[1], d1[10], "Additional dataset random samples", 2, xl='Sample 1', yl='Sample 2')

with np.load('model1_res_valid.npz') as f:
	d1 = f['arr_0'] 
	print(np.shape(d1))
	d2 = f['arr_3'] 
	print(np.shape(d2)) 
	plot(d1[1], d2[1], "Additional dataset valid result", 4)

# with np.load('multi_sets2/model1_res_control.npz') as f:
# 	data2_in= f['arr_0']

# with np.load('multi_sets2/model0_res_control.npz') as f:
# 	data1_in= f['arr_0']


# with np.load('multi_sets2/model2_res_train.npz') as f:
# 	d1= f['arr_0']
# 	d2 = f['arr_1']

# plot(d2[0], d1[0], 'dataset 3 train result', 5)

# with np.load('multi_sets2/model2_res_decoded.npz') as f:
# 	d1= f['arr_0']

# plot(d1, data2_in, 'cross set', 6, xl='3 set\'s sample decoded by 2\'s decoder', yl='2\'s own sample')
plt.show()