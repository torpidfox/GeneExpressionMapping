import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA




with np.load('model0_res_valid.npz') as f:
	model0 = f['arr_2']

with np.load('model1_res_valid.npz') as f:
	model1 = f['arr_2']

# t1 = TSNE(n_components=3).fit_transform(model0)

# t2 = TSNE(n_components=3).fit_transform(model1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter([p[0] for p in t1], [p[1] for p in t1], [p[2] for p in t1])
# ax.scatter([p[0] for p in t2], [p[1] for p in t2], [p[2] for p in t2])
# plt.show()

pca = PCA(n_components=3)
x1 = pca.fit_transform(model0)
x2 = pca.fit_transform(model1)


ax.scatter([p[0] for p in x1],
	[p[1] for p in x1],
	[p[2] for p in x1])
ax.scatter([p[0] for p in x2[:25]], [p[1] for p in x2[:25]], [p[2] for p in x2[:25]], label='2nd set schiz')
ax.scatter([p[0] for p in x2[25:]], [p[1] for p in x2[25:]], [p[2] for p in x2[25:]], label='2nd set control')
plt.title('Squezeed data')
plt.legend()
plt.show()