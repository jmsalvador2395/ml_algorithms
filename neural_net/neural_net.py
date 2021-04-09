import numpy as np

def data_to_array(fname):
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

	return trng_labels, trng_set

class TwoLayerNN:
	def __init__(self, num_inputs=1, num_outputs=1):
		#instantiate number of inputs
		self.num_inputs=num_inputs

		#instanstiate number of outputs
		self.num_outputs=num_outputs

		#100 hidden units for input layer
		self.h_size=100

		#randomly initialize hidden layer weights
		self.h_matrix=np.random.randn(self.h_size, num_inputs+1) #+1 on num_inputs for the bias

		#set bias weight to 1 for input weight vectors
		self.h_matrix[:,0]=1

		#randomly initialize output layer weights
		self.v_matrix=np.random.randn(num_outputs, self.h_size+1) #+1 on h_size to account for bias

		#set bias weight to 1 for output weight vectors
		self.v_matrix[:,0]=1


	def train(fname, shape):
		trng_labels, trng_set = data_to_array(fname)



	def forward_propogation(self, data):
		data=np.insert(data, 0, 1.0, axis=0)
		#compute dot prodcuts for data with all hidden layer vectors into hidden_output
		hidden_output=np.zeros(self.h_size+1)
		for i in range(self.h_matrix.shape[0]):
			hidden_output[i]=np.dot(self.h_matrix[i], data)

		#comput dot products for hidden_output with output layer vectors into output
		output=np.zeros(self.num_outputs)
		for i in range(self.v_matrix.shape[0]):
			output[i]=np.dot(self.v_matrix[i], hidden_output)


		return output
