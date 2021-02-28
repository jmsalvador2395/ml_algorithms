import heapq
import numpy as np
from sklearn.datasets import load_svmlight_file


def knn_manhattan(k, fname_tr, fname_t, num_features):
	tr_data=load_svmlight_file(fname_tr)
	t_data=load_svmlight_file(fname_t)

	print("shapes: " + str(tr_data[0].get_shape()) + " " + str(t_data[0].get_shape()))

	if not tr_data[0].get_shape()[1] == num_features:
		tr_data[0].reshape((tr_data[0].get_shape()[0], num_features))
	if not t_data[0].get_shape()[1] == num_features:
		t_data[0].reshape((t_data[0].get_shape()[0], num_features))
		


	for i in range(t_data[0].get_shape()[0]): #loop to iterate through each test sample
		neighbors=[]
		xi=t_data[0].getrow(i).toarray()[0]
		for j in range(tr_data[0].get_shape()[0]): #loop to iterate through each training sample
			y=tr_data[0].getrow(j).toarray()[0]
			distance=np.sum(np.absolute(y-xi)) #calculate manhattan distance
			if(len(neighbors)<k):
				heapq.heappush(neighbors, (distance, tr_data[1][j]))
				heapq._heapify_max(neighbors)
			else:
				if(neighbors[0]):
					heapq.heapreplace(neighbors,(distance, tr_data[1][j]))
					heapq._heapify_max(neighbors)
		neighbor_labels=[x[1] for x in neighbors]
		print(str(max(set(neighbor_labels),key=neighbor_labels.count)))

def knn_euclidean(k, fname_tr, fname_t, num_features):
	tr_data=load_svmlight_file(fname_tr)
	t_data=load_svmlight_file(fname_t)

	print("shapes: " + str(tr_data[0].get_shape()) + " " + str(t_data[0].get_shape()))
	if not tr_data[0].get_shape()[1] == num_features:
		tr_data[0].reshape((tr_data[0].get_shape()[0], num_features))
	if not t_data[0].get_shape()[1] == num_features:
		t_data[0].reshape((t_data[0].get_shape()[0], num_features))
		
	for i in range(t_data[0].get_shape()[0]): #loop to iterate through each test sample
		neighbors=[]
		xi=t_data[0].getrow(i).toarray()[0]
		for j in range(tr_data[0].get_shape()[0]): #loop to iterate through each training sample
			y=tr_data[0].getrow(j).toarray()[0]
			distance=np.sqrt(np.sum(np.square(y-xi))) #calculate euclidean distance
			if(len(neighbors)<k):
				heapq.heappush(neighbors, (distance, tr_data[1][j]))
				heapq._heapify_max(neighbors)
			else:
				if(neighbors[0]):
					heapq.heapreplace(neighbors,(distance, tr_data[1][j]))
					heapq._heapify_max(neighbors)
		neighbor_labels=[x[1] for x in neighbors]
		print(str(max(set(neighbor_labels),key=neighbor_labels.count)))


