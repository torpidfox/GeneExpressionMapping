import matplotlib.pyplot as mp 
import numpy as np 
from sklearn.manifold import TSNE 

def plot(x, y, title, pos, xl='Original', yl='Decoded'): 
	sub_coord = 220 + pos 
	ax = mp.subplot(sub_coord) 
	mse = ((x - y) ** 2).mean()
	mp.scatter(x, y, label=str(mse))
	mp.xlabel(xl) 
	mp.ylabel(yl) 
	mp.title(title) 
	mp.legend(loc=2)
	# mp.xlim(left=0, right=14)
	# mp.ylim(bottom=0, top=14)
	mp.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".1") 

with np.load('model0_res_valid.npz') as f: 
	d1 = f['arr_0'] 
	print(np.shape(d1))
	d2_1 = f['arr_3'] 
	plot(d2_1[13], d1[13], "Main dataset valid result", 3)


with np.load('model0_res_valid.npz') as f:
	d1 = f['arr_0'] 
	print(np.shape(d1))
	d2 = f['arr_3'] 
	print(np.shape(d2)) 
	plot(d1[9], d1[18], "Main dataset random samples", 1, xl='Sample 1', yl='Sample 2')  

# with np.load('model1_res_valid.npz') as f: 
# 	d1 = f['arr_0'] 
# 	print(np.shape(d1))
# 	d2 = f['arr_1'] 
# 	print(np.shape(d2)) 
# 	plot(d2[1], d1[0], "Additional dataset random samples", 2)

# with np.load('model1_res_valid.npz') as f:
# 	d1 = f['arr_0'] 
# 	print(np.shape(d1))
# 	d2 = f['arr_1'] 
# 	print(np.shape(d2)) 
# 	plot(d2[0], d1[0], "Additional dataset valid result", 4)

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


mp.show()