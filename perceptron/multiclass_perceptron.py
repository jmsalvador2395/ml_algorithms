import numpy as np
import math
import perceptron

from sklearn.datasets import load_svmlight_file
#data=load_svmlight_file(sys.argv[1])

"""
x is the test data

w is the weight vector

bias is the offset or threshold. No need to set this if you used the
learning() function

returns: 1 for class 1 and -1 for class 2
"""
def testset_activation(fname, w, num_labels, start_label, bias=0):
	data=load_svmlight_file(fname)
	num_samples=data[0].get_shape()[0]
	print("num samples: " + str(num_samples))
	for i in range(num_samples):
		xi=np.insert(data[0].getrow(i).toarray()[0],0,1)
		label=start_label+num_labels-1
		current_label=start_label
		for j in range(w.shape[0]):
			y_hat=np.dot(xi,w[j])
			if(y_hat<=current_label):
				label=current_label
				break
			current_label+=1
		print("test data " + str(i) + ": " + str(label)
				+ "\ttrue label: " + str(data[1][i]))
		
	

"""
x is the training set which is an NxD matrix

y is the 1xD set of labels corresponding to the training set
y is either 1 or -1

limit is factor of 100 for multiples of training iterations.
limit=1 means there will be 10000 iterations, 2 will be 2000 and so on
This is used because the training process doesn't converge for non-linearly
separable data

step size is a constant used for updating the weights

returns: weight vector to use as input on activation function
"""
def learning(fname, limit,num_labels, start_label, step_size, bias=0):
	data=load_svmlight_file(fname)
	samples=data[0].get_shape()[0]
	features=data[0].get_shape()[1]
	current_label=start_label
	for j in range(num_labels-1):#need to do more with this
		limit_count=0
		temp_w=np.ones(features+1)
		while limit_count < limit*1000:
			y_hat=0
			count=0	#counts the amount of updates 
			for i in range(samples):
				if data[1][i] >= current_label:
					xi=np.insert(data[0].getrow(i).toarray()[0],0,1)
					y_hat=np.dot(temp_w,xi)

					#use this to separate labels into 2 classes
					if data[1][i]-current_label >=1:
						true_label=1
					else:
						true_label=-1

					if y_hat-bias > 0:
						y_hat=1
					else:
						y_hat=-1
					if not y_hat == true_label:	#data[1][i] is the label
						temp_w=temp_w+step_size*true_label*xi
						count+=1
			if count==0:#check if w has converged and break if yes
				print("weight vector " + str(j) + " converged!")
				break
			limit_count+=1
		current_label+=1
		if limit_count == limit*1000:
			print("weight vector " + str(j) + " did not converge")
		if j == 0:
			w=np.array([temp_w])
		else:
			w=np.vstack((w, temp_w))
	return w
