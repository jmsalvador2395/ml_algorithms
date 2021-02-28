import numpy as np
from scipy.sparse.linalg import eigs

def pca(x, k_eigs):

	#mean of data
	mean=np.mean(x, axis=1)

	#number of data
	n=x.shape[0]

	#convert to zero mean
	x=x-np.tile(mean, (x.shape[1],1)).transpose()

	#create covariance matrix
	cov_x=np.cov(x)

	#compute eigenvalues
	lmda, u=eigs(cov_x, k=k_eigs)

	#sort by top eigenvalues
	eig_pairs=[(np.abs(lmda[i]), u[:,i]) for i in range(len(lmda))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)


	#split back into respective variables
	u=np.array([n[1] for n in eig_pairs])
	lmda=np.array([n[0] for n in eig_pairs])

	#compute embeddings
	e=u.dot(x)

	return e, u, lmda


