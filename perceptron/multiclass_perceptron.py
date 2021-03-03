import numpy as np
import math
import perceptron
import re

from sklearn.datasets import load_svmlight_file
#data=load_svmlight_file(sys.argv[1])

"""
takes libsvm/svmlight file to test weight vector on

fname is the file name

w is the weight vector

bias is the offset or threshold. No need to set this if you used the
learning() function

prints out prediction statistics
"""
def activation(fname, w, features, bias=0):

	#iterate through file once to get the label set
	label_set=[]
	d_reader=open(fname, 'r')
	for line in d_reader:
		label=float(re.split(" ",line)[0])
		if not label in label_set:
			label_set.append(label)
	d_reader.close()
	label_set.sort()
	#finished reading in label set

	#iterate through test set
	d_reader=open(fname, 'r')

	total=0
	class_stats={}
	for line in d_reader:
		#process line into numpy array xi
		xi=np.zeros(features+1) #+1 for prepended 1
		xi[1]=1.
		ln_split=re.split(" ",line)
		true_label=int(ln_split[0]) #read true label
		for j in ln_split[1:-1]:
			tup=re.split(":",j)
			xi[int(tup[0])]=float(tup[1]) #implied +1 and -1 to take prepended 1 and array representations into account
		#xi is now usable

		#generate guess
		for i in range(len(label_set)-1):
			if np.dot(w[i],xi)<=0:
				y_hat=label_set[i]
				break
			else:
				y_hat=label_set[i+1]
		#count up results
		if true_label==y_hat:
			if not true_label in class_stats:
				class_stats[true_label]=[0,0]
			class_stats[true_label][0]+=1
		else:
			if not true_label in class_stats:
				class_stats[true_label]=[0,0]
			class_stats[true_label][1]+=1
		total+=1
	#print results
	print("**" +str(total) + " total test samples**")
	for key in sorted(class_stats):
		print("stats for label " + str(key))
		print(str(class_stats[key][0]) + " correct classifications")
		print(str(class_stats[key][1]) + " misclassifications to " + str(key))
		print()
	total_misclassified=sum(class_stats[key][1] for key in class_stats)
	total_correct_classified=sum(class_stats[key][0] for key in class_stats)
	print(str(total_misclassified) + " out of " + str(total)
			+ " misclassified\nAccuracy: " 
			+ "{:.2f}".format(100.*total_correct_classified/total)
			+ "%")
		
	

"""
takes in a libsvm/svmlight file to train weight vectors on

fname is the file name


limit is factor of 100 for multiples of training iterations.
limit=1 means there will be 100 iterations, 2 will be 2000 and so on
This is used because the training process doesn't converge for non-linearly
separable data

step size is a constant used for updating the weights

returns: weight vector to use as input on activation function
"""
def learning(fname, limit, features, step_size, bias=0):

	#iterate through file once to get the label set
	label_set=[]
	d_reader=open(fname, 'r')
	for line in d_reader:
		label=float(re.split(" ",line)[0])
		if not label in label_set:
			label_set.append(label)
	d_reader.close()
	label_set.sort()
	#finished reading in label set

	w=np.ones((len(label_set)-1,features+1)) #instantiate weights

	for i in range(len(label_set)-1):
		#process line into numpy array xi
		limit_count=0
		while limit_count < limit*100:
			d_reader=open(fname, 'r')
			count=0 #used to check if w converged
			for line in d_reader:
				ln_split=re.split(" ",line)
				true_label=int(ln_split[0]) #read true label
				if true_label >= label_set[i]: #processing gets skipped if weight is already generated for this lab
					xi=np.zeros(features+1)
					xi[0]=1
					for j in ln_split[1:-1]: 
						tup=re.split(":",j)
						xi[int(tup[0])]=float(tup[1])
					#xi now usable

					y_hat=np.dot(w[i],xi) #make guess

					#map true label and guess label to 1 or -1
					if true_label > label_set[i]:
						mapped_label=1
					else:
						mapped_label=-1
					#map y_hat calculation to 1 or -1
					if y_hat-bias > 0:
						y_hat=1
					else:
						y_hat=-1
					#end mapping

					if not (y_hat == mapped_label): #check if w needs to be updated
						w[i]=w[i]+step_size*mapped_label*xi #update w
						count+=1
			if count == 0:
				print("weight vector " + str(i+1) + " converged!")
				break
			limit_count+=1
			d_reader.close()
		if limit_count == limit*100:
			print("weight vector " + str(i+1) + " did not converge")
	return w
		

