import numpy as np
import random

"""
implemented for numpy arrays

trng_set is a TxD numpy array
labels is a 1d numpy array
lmda is the scalar
T is the amount of samples
"""
def pegasos(trng_set, labels,  lmda, T):
	features=trng_set.shape[1] #column size of numpy array
	w=np.zeros(features+1)

	rnd_range=list(range(T)) #create list range of numbers
	random.shuffle(rnd_range) #scramble the list

	for t in range(1,T):
		nt=1/(lmda*t) #calculate step size
		if labels[rnd_range[t]]*np.dot(w, np.insert(trng_set[rnd_range[t]], 0, 0)) < 1:
			w=(1-nt*lmda)*w+(nt*labels[rnd_range[t]]*np.insert(trng_set[rnd_range[t]], 0, 0))
		elif labels[rnd_range[t]]*np.dot(w[t], trng_set[rnd_range[t]]) >= 1:
			w=(1-nt*lmda)*w
	return w
