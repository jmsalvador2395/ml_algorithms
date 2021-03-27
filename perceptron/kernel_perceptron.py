import numpy as np
import re

"""
fname_t is the test data

fname_tr is the training data

alpha is the weight vector of size N

features is the number of dimensions for each sample

a and b are hyperparameters of the kernel function

d is the polynomial of the kernel function


returns: 1 for class 1 and -1 for class 2
"""
def poly_kernel_activation(fname_t,
						   fname_tr,
						   alpha,
						   features,
						   a=0,		#hyperparameter of kernel function
						   b=1,		#hyperparameter of kernel function
						   d=1):	#polynomial of kernel function
	trng_set_file=open(fname_tr, 'r')
	
	#iterate through the file once to count the number of samples and read label set
	trng_samples=0
	d_reader=open(fname_tr, 'r')
	for line in d_reader:
		trng_samples+=1
		label=int(re.split(" ",line)[0])
	d_reader.close()
	
	#finish counting and reading labels



	"""
	read training data into numpy array
	this is probably a memory hog but it's
	faster than iterating through the file thousands of times
	from the hard drive
	"""
	row=0
	trng_set=np.zeros((trng_samples, features+1))
	trng_labels=np.zeros(trng_samples)
	for line in trng_set_file:
		#process line into xi
		ln_split=re.split(" ",line)
		trng_labels[row]=int(ln_split[0])
		xi=np.zeros(features+1)
		xi[0]=1
		for j in ln_split[1:-1]:
			tup=re.split(":",j)
			xi[int(tup[0])]=float(tup[1])
		#xi now usable
		for j in range(features+1):
			trng_set[row][j]=xi[j]
		row+=1
	trng_set_file.close()
	#end reading training data



	d_reader=open(fname_t, 'r')

	total=0
	true_positive=0
	true_negative=0
	false_positive=0
	false_negative=0

	i=0
	for line in d_reader:
		xi=np.zeros(features+1) #+1 for prepended 1
		xi[1]=1.
		ln_split=re.split(" ",line)
		true_label=int(ln_split[0])
		for j in ln_split[1:-1]:
			tup=re.split(":",j)
			xi[int(tup[0])]=float(tup[1]) #+1 to take prepended 1 into account but then -1 because indeces start from 1
		#compute guess using kernel method
		if np.sum(alpha[n]*(a+b*np.dot(trng_set[n],trng_set[i])**d) for n in range(trng_set.shape[0])) > 0:
			if true_label==1:
				true_positive+=1
			else:
				false_positive+=1
		else:
			if true_label==-1:
				true_negative+=1
			else:
				false_negative+=1
		total+=1
		i+=1
	print("\ntrue positives: " + str(true_positive)
				+ "\ntrue negatives: " + str(true_negative)
				+ "\nfalse positives: " + str(false_positive)
				+ "\nfalse negatives: " + str(false_negative))
	print("\n" + str(false_positive+false_negative) + " out of "
				+ str(total) + " misclassified\nAccuracy: "
				+ "{:2f}".format(100.*(true_positive+true_negative)/total)
				+ "%")
"""
x is the training set which is an NxD matrix

y is the 1xD set of labels corresponding to the training set
y is either 1 or -1

limit is factor of 100 for multiples of training iterations.
limit=1 means there will be 100 iterations, 2 will be 2000 and so on
This is used because the training process doesn't converge for non-linearly
separable data

step size is a constant used for updating the weights

returns: weight vector to use as input on activation function
"""
def poly_kernel_learn(fname,
					  fname_tr,
					  limit,
					  features,
					  step_size=1,
					  a=0,	#hyperparameter of kernel function
					  b=1,	#hyperparameter of kernel function
					  d=1):	#polynomialof kernel function
	trng_set_file=open(fname_tr, 'r')
	
	#iterate through the file once to count the number of samples and read label set
	trng_samples=0
	d_reader=open(fname_tr, 'r')
	for line in d_reader:
		trng_samples+=1
		label=int(re.split(" ",line)[0])
	d_reader.close()
	
	#finish counting and reading labels



	"""
	read training data into numpy array
	this is probably a memory hog but it's
	faster than iterating through the file thousands of times
	from the hard drive
	"""
	row=0
	trng_set=np.zeros((trng_samples, features+1))
	trng_labels=np.zeros(trng_samples)
	for line in trng_set_file:
		#process line into xi
		ln_split=re.split(" ",line)
		trng_labels[row]=int(ln_split[0])
		xi=np.zeros(features+1)
		xi[0]=1
		for j in ln_split[1:-1]:
			tup=re.split(":",j)
			xi[int(tup[0])]=float(tup[1])
		#xi now usable
		for j in range(features+1):
			trng_set[row][j]=xi[j]
		row+=1
	trng_set_file.close()
	#end reading training data



	alpha=np.ones(trng_samples)
	j=0
	while j<limit*100:
		count=0 #counts the number of updates
		d_reader=open(fname, 'r')
		for i in range(trng_set.shape[0]):
			#compute y_hat using kernel method
			y_hat=np.sum(alpha[n]*(a+b*np.dot(trng_set[n],trng_set[i])**d) for n in range(trng_set.shape[0]))
			if y_hat>0:
				y_hat=1
			else:
				y_hat=-1
			if not (trng_labels[i] == y_hat): #check if guess is wrong
				alpha[i]=alpha[i]+step_size*trng_labels[i] #make corrections
				count+=1
		if count==0:
			print("converged!")
			break
		j+=1
	if j == limit*100:
		print("did not converge")
	return alpha
