import heapq
import numpy as np
import re
import datetime


def knn(k, fname_tr, fname_t, features, metric):
	trng_set_file=open(fname_tr, 'r')

	
	#iterate through the file once to count the number of samples
	trng_samples=0
	d_reader=open(fname_tr, 'r')
	for line in d_reader:
		trng_samples+=1
	d_reader.close()
	

	trng_set=np.zeros((trng_samples, features))
	trng_labels=np.zeros(trng_samples)

	"""
	read training data into numpy array
	this is probably a memory hog but it's
	faster than iterating through the file thousands of times
	from the hard drive
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
	trng_set_file.close()
	#end reading training data


	row=0 #i am also using this to keep track of the total amount of test samples
	test_set_file=open(fname_t,'r')
	start_time=datetime.datetime.now()
	class_stats={}
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

		counters={}
		for i in neighbor_labels:
			if i not in counters:
				counters[i]=0
			counters[i]+=1
		y_hat=max(counters, key=counters.get)
			
		if y_hat == true_label:
			if not true_label in class_stats:
				class_stats[true_label]=[0,0]
			class_stats[true_label][0]+=1

		else:
			if not true_label in class_stats:
				class_stats[true_label]=[0,0]
			class_stats[true_label][1]+=1
		row+=1
	test_set_file.close()
	total_time=datetime.datetime.now()-start_time
	print("**"+str(row) + " total test samples**")
	print("avg time to compute " + str(k)
				+ " neighbors: "
				+ "{:.2f}".format(float(total_time.total_seconds())/row)
				+ " seconds per sample\n")
	for key in sorted(class_stats):
		print("stats for label " + str(key))
		print(str(class_stats[key][0]) + " correct classifications")
		print(str(class_stats[key][1]) + " misclassifications to " + str(key))
		print()
	total_misclassified=sum(class_stats[key][1] for key in class_stats)
	total_correct_classified=sum(class_stats[key][0] for key in class_stats)
	print("\n" + str(total_misclassified)
				+ " out of " + str(row) + " misclassified\n"
				+ "Accuracy: " + "{:.2f}".format(100.*total_correct_classified/row))
		
		
		

