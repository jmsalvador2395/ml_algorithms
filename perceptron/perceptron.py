import numpy as np
import re

"""
x is the test data

w is the weight vector

bias is the offset or threshold. No need to set this if you used the
learning() function

returns: 1 for class 1 and -1 for class 2
"""
def activation(fname, w, features, bias=0):
	d_reader=open(fname, 'r')

	for line in d_reader:
		xi=np.zeros(features+1) #+1 for prepended 1
		xi[1]=1.
		ln_split=re.split(" ",line)
		true_label=float(ln_split[0])
		for j in ln_split[1:-1]:
			tup=re.split(":",j)
			xi[int(tup[0])]=float(tup[1]) #+1 to take prepended 1 into account but then -1 because indeces start from 1
		if np.dot(xi, w)-bias>0:
			if int(true_label)==1:
				print("true positive")
			else:
				print("false positive")
		else:
			if int(true_label)==1:
				print("true negative")
			else:
				print("false negative")
"""
x is the training set which is an NxD matrix

y is the 1xD set of labels corresponding to the training set
y is either 1 or -1

limit is factor of 1000 for multiples of training iterations.
limit=1 means there will be 1000 iterations, 2 will be 2000 and so on
This is used because the training process doesn't converge for non-linearly
separable data

step size is a constant used for updating the weights

returns: weight vector to use as input on activation function
"""
def learning(fname, limit, features, step_size, bias=0):
	"""
	data=load_svmlight_file(fname)
	sample_size=data[0].get_shape()[0]
	read_features=data[0].get_shape()[1]
	"""

	w=np.ones(features+1)
	i=0
	while i<limit*1000:
		count=0 #counts the number of updates
		d_reader=open(fname, 'r')
		for line in d_reader:
			#process line into numpy array xi
			xi=np.zeros(features+1)
			xi[0]=1 #i forget if i need this
			ln_split=re.split(" ",line)
			true_label=float(ln_split[0])
			for j in ln_split[1:-1]:
				tup=re.split(":",j)
				xi[int(tup[0])]=float(tup[1])
			#xi now usable
			
			y_hat=np.dot(w,xi) #make guess
			if y_hat-bias>0:
				y_hat=1
			else:
				y_hat=-1
			if not true_label == y_hat: #check if guess is wrong
				w=w+step_size*true_label*xi #make corrections
				count+=1
		if count==0:
			print("converged!")
			break
		i+=1
	if i == limit*1000:
		print("did not converge")
	return w
