import heapq
import numpy as np
import re
from sklearn.datasets import load_svmlight_file


def knn(k, fname_tr, fname_t, trng_samples, features, metric):
	trng_set_file=open(fname_tr, 'r')

	trng_set=np.zeros((trng_samples, features))
	trng_labels=np.zeros(trng_samples)

	"""
	read training data into numpy array
	this is probably a memory hog but it's
	faster than iterating through the file thousands of times
	"""
	row=0
	for line in trng_set_file:
		#process line into xi
		ln_split=re.split(" ",line)
		trng_labels[row]=float(ln_split[0])
		xi=np.zeros(features)
		for j in ln_split[1:-1]:
			tup=re.split(":",j)
			xi[int(tup[0])-1]=float(tup[1])
		#xi now usable
		for j in range(features):
			trng_set[row][j]=xi[j]
		row+=1
	#end reading training data
	row=0
	test_set_file=open(fname_t,'r')
	for line in test_set_file:
		neighbors=[]
		#processing to read training sample into numpy array
		ln_split=re.split(" ",line)
		true_label=float(ln_split[0])
		tst_smpl=np.zeros(features)
		for j in ln_split[1:-1]:
			tup=re.split(":",j)
			tst_smpl[int(tup[0])-1]=float(tup[1])
		#test sample now usable

		for i in range(trng_samples):
			if metric == 'manhattan':
				distance=np.sum(np.absolute(tst_smpl-trng_set[i])) #calculate manhattan distance
			elif metric == 'euclidean':
				distance=np.sqrt(np.sum(np.square(tst_smpl-trng_set[i]))) #calculate euclidean distance
			
			if(len(neighbors)<k):
				heapq.heappush(neighbors, (distance, trng_labels[i]))
			else:
				if((distance, trng_labels[i])<neighbors[0]):
					heapq.heapreplace(neighbors, (distance, trng_labels[i]))
					heapq._heapify_max(neighbors)
		neighbor_labels=[x[1] for x in neighbors]
		"""
		if max(set(neighbor_labels),key=neighbor_labels.count) ==  true_label:
			print("line " + str(row+1) + ": " + str(max(set(neighbor_labels),key=neighbor_labels.count)) + " positive")
		else:
			print("line " + str(row+1) + ": " + str(max(set(neighbor_labels),key=neighbor_labels.count)) + " negative")
		"""
		if max(set(neighbor_labels),key=neighbor_labels.count) ==  true_label:
			print("line " + str(row+1) + ": " + str(true_label) + " positive")
		else:
			print("line " + str(row+1) + ": " + str(true_label) + " negative")
		row+=1

