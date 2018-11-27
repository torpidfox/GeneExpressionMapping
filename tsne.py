import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA




with np.load('model0.0_res_shared_in.npz') as f:
	model0 = f['arr_0']
	model1 = f['arr_1']

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
ax.scatter([p[0] for p in x2], [p[1] for p in x2], [p[2] for p in x2])
plt.show()