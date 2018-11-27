from sklearn.svm import SVC
from random import shuffle
import pandas as pd	

def import_data(inds):
	data = list()
	#all_data = list()
	i = 0

	for filename in ["../data/batched/batch{}.txt".format(i) for i in range(26)]:
		with open(filename) as f:			
			for l in f:
				sample = l.split()
				#all_data.append(list(map(float, sample)))

				if i in inds:
					data.append(list(map(float, sample)))

				i+=1

	return data


df = pd.read_csv('../data/E-MTAB-3732.sdrf.txt',
	sep='	')

keys = ['brain', 'blood']

inds = list()
tags = list()

for i, key in enumerate(keys):
	key_list = list(df.index[df["Characteristics[organism part]"] == key])
	inds += key_list

	for el in key_list:
		tags.append(i)

data = import_data(inds)
tagged = list(zip(data, tags))
shuffle(tagged)
data[:], tags[:] = zip(*tagged)

clf = SVC()
clf.fit(data[:len(data) // 2], tags[:len(data) // 2])
print(clf.score(data[len(data) // 2:], tags[len(data) //2 : ]))



