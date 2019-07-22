import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from random import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


num = 2
count = 50

def read(filenames, arr=1, tags=[0, 1]):

	with np.load(filenames[0]) as f:
		print(np.shape(f['arr_{}'.format(arr)]))
		data1_in = f['arr_{}'.format(arr)][:count]
		t1 = [tags[0]] * count
		shape = [0, len(data1_in)]

	with np.load(filenames[1]) as f:
		print(np.shape(f['arr_{}'.format(arr)]))
		data1_in = np.append(data1_in, f['arr_{}'.format(arr)][:count], axis=0)
		t1 += [tags[0]] * len(f['arr_{}'.format(arr)])
		shape += [shape[-1] + count]

	with np.load(filenames[2]) as f:
		print(np.shape(f['arr_{}'.format(arr)][:20]))
		data1_in = np.append(data1_in, f['arr_{}'.format(arr)], axis=0)
		t1 += [tags[0]] * len(f['arr_{}'.format(arr)][:20])
		shape += [shape[-1] + 20]

	with np.load(filenames[3]) as f:
		print(np.shape(f['arr_{}'.format(arr)]))
		data1_in = np.append(data1_in, f['arr_{}'.format(arr)][:count], axis=0)
		t1 += [tags[1]] * len(f['arr_{}'.format(arr)])
		shape += [shape[-1] + count]

	# with np.load(filenames[4]) as f:
	# 	print(np.shape(f['arr_{}'.format(arr)]))
	# 	data1_in = np.append(data1_in, f['arr_{}'.format(arr)], axis=0)
	# 	t1 += [tags[1]] * len(data1_in)
	# 	shape += [shape[-1] + count]

	# with np.load(filenames[5]) as f:
	# 	print(np.shape(f['arr_{}'.format(arr)]))
	# 	data1_in = np.append(data1_in, f['arr_{}'.format(arr)][:20], axis=0)
	# 	t1 += [tags[1]] * 20
	# 	shape += [shape[-1] + 20]

	return data1_in, t1, shape

def fit(data):
	t = TSNE(n_components=3)
	transformed = t.fit_transform(data)

	return transformed

def draw(dots, t, c, m='.'):
	ax.scatter([p[0] for p in dots], [p[1] for p in dots], [p[2] for p in dots],label=t, marker=m, color=c)

def draw_multi(dots, borders, tags):
	col = ['r'] * num + ['g'] * num
	print(col)
	sym = ['.' if i % num == 0 else '>' if i % num == 1 else '<' for i in range(num*2)]

	for i, t in enumerate(tags):
		draw(dots[borders[i]:borders[i+1]], t, col[i], sym[i])


f = ['multi_sets2/model{}_res_control_squeezed.npz'.format(i) for i in range(num)]
f += ['multi_sets2/model{}_res_schiz_squeezed.npz'.format(i) for i in range(num)]



tags = ['set {} control'.format(i) for i in range(num)]
tags += ['set {} schiz'.format(i) for i in range(num)]

data, t, b = read(f)
# with np.load('multi_sets2/model2_res_bipolar3_squeezed.npz') as f:
# 	data = np.append(data, f['arr_1'][:20], axis=0)
# 	b += [b[-1] + 20]

# with np.load('multi_sets2/model2_res_deprsession3_squeezed.npz') as f:
# 	data = np.append(data, f['arr_1'][:20], axis=0)
# 	b += [b[-1] + 20]

# with np.load('multi_sets2/model0_res_bipolar3_squeezed.npz') as f:
# 	data = np.append(data, f['arr_1'][:count], axis=0)
# 	b += [b[-1] + count]

tr = fit(data)
print(np.shape(data))
print(len(data))
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')
print(b)
draw_multi(tr, b, tags)
#ax.scatter([p[0] for p in data[-90:-70]], [p[1] for p in data[-90:-70]], [p[2] for p in data[-90:-70]],label='set 3 bipolar', marker='.', color='b')
#ax.scatter([p[0] for p in data[-70:-count]], [p[1] for p in data[-70:-50]], [p[2] for p in data[-70:-50]],label='set 3 depression', marker='.', color='c')
#ax.scatter([p[0] for p in data[-50:]], [p[1] for p in data[-50:]], [p[2] for p in data[-50:]],label='set 2 bipolar', marker='>', color='b')

plt.legend(loc=4)
plt.title('Inner layer, svm accuracy 0.8')
tagged = list(zip(data, t))
shuffle(tagged)
clf = SVC()
t = list(zip(*tagged))
clf.fit(t[0][:len(tagged) // 2], t[1][:len(tagged) // 2])
print(clf.score(t[0][len(tagged) // 2:], t[1][len(tagged) // 2:]))

plt.show()
# f = ['multi_sets2/model{}_res_control_decoded.npz'.format(i) for i in range(3)]
# f += ['multi_sets2/model{}_res_schiz_decoded.npz'.format(i) for i in range(3)]

# data, t, b = read(f, 0)

# with np.load('multi_sets2/model0_res_bipolar3_decoded.npz') as f:
# 	print(np.shape(f['arr_0']))
# 	data = np.append(data, f['arr_0'][:20], axis=0)
# 	b += [b[-1] + 20]

# with np.load('multi_sets2/model0_res_deprsession3_decoded.npz') as f:
# 	data = np.append(data, f['arr_0'][:20], axis=0)
# 	b += [b[-1] + 20]

# print(np.shape(data))
# print(len(data))
# tr = fit(data)
# print(len(data))
# print(b)
# ax = fig.add_subplot(111, projection='3d')
# draw_multi(tr, b, tags)
# ax.scatter([p[0] for p in data[-40:-20]], [p[1] for p in data[-40:-20]], [p[2] for p in data[-40:-20]],label='set 3 bipolar', marker='.', color='c')
# ax.scatter([p[0] for p in data[-20:]], [p[1] for p in data[-20:]], [p[2] for p in data[-20:]],label='set 3 depression', marker='.', color='k')
# plt.legend(loc=4)
# plt.title('Different datasets\' samples decoded by the same decoder, svm accuracy 0.6')
# plt.show()


# tagged = list(zip(data, t))
# shuffle(tagged)
# clf = SVC()
# t = list(zip(*tagged))
# clf.fit(t[0][:len(tagged) // 2], t[1][:len(tagged) // 2])
# print(clf.score(t[0][len(tagged) // 2:], t[1][len(tagged) // 2:]))
