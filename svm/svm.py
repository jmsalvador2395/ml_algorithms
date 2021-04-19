import numpy as np
import cupy as cp
import re
import math
from math import log
import re
from datetime import datetime as dt
import random
 
"""
helper function to read in the training data
"""
def data_to_array(fname, features, trng_samples):
        print("*** begin reading data ***")
        trng_set=np.zeros((trng_samples, features))
        trng_labels=np.zeros(trng_samples)

        begin=dt.now()

        """
        read training data into numpy array
        this is probably a memory hog but it's
        faster than iterating through the file thousands of times
        from the hard drive
        """
        row=0
        trng_set_file=open(fname, 'r')
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
        elapsed=dt.now()-begin
        print("*** it took", elapsed.total_seconds(), "to read in the data ***")
        #end reading training data
        
        return trng_labels, trng_set

  
"""
converts array of binary digits into decimal number
"""
def bin_str_to_int(input_string):
    return int(np.array2string(cp.asnumpy(input_string[::-1]).astype(np.int32), precision=0, separator='')[1:-1], 2)

"""
converts integer into list of 0s and 1s
index 0 is the least significant bit
"""
def integer_to_bit_list(input_int, size):
    return[int(input_int) >> i&1 for i in range(size)]

"""
returns a or b depending on which is smaller
"""
def min(a, b):
    if a<b:
        return a
    else:
        return b


class SVM:
    def __init__(self, features, fname_t, trng_samples, label_set):
        
        #number of features
        self.features=features

        #label set
        self.label_set=label_set

        #read in trng set
        self.trng_labels, self.trng_set=data_to_array(fname_t, self.features, trng_samples)

        #number of classes
        self.num_classes=len(label_set)

        #number of classifiers
        self.num_classifiers=int(math.ceil(log(self.num_classes, 2)))

        #initialize weight vectors
        self.W=np.zeros((self.num_classifiers, features+1))


    """
    implemented for numpy arrays

    trng_set is a TxD numpy array
    labels is a 1d numpy array
    lmda is the scalar
    T is the amount of samples
    """
    def pegasos(self, lmda, limit):
        w=np.zeros(self.features+1)

        T=self.trng_set.shape[0]

        rnd_range=list(range(T)) #create list range of numbers
        random.seed()
        random.shuffle(rnd_range) #scramble the list

        for j in range(limit*100):
            for t in range(T):
                nt=1/(lmda*(1+t)) #calculate step size

                #prepend bias feature to xi and transpose
                xi=np.insert(self.trng_set[rnd_range[t]], 0, 1)

                #make prediction
                y_hat_index, y_hat_ecoc_raw=self.prediction(xi)

                #take true label and separate it into array of bits
                true_label_index=self.label_set.index(self.trng_labels[rnd_range[t]])
                true_label_bits=cp.array(integer_to_bit_list(true_label_index, self.num_classifiers))

                #convert 0s in true_label_bits to -1
                true_label_bits[true_label_bits == 0] = -1

                for i in range(self.num_classifiers):
                    if true_label_bits[i]*y_hat_ecoc_raw[i] < 1:
                        (1-nt*lmda)*self.W[i]
                        xi*(nt*int(true_label_bits[i]))

                        self.W[i]=((1-nt*lmda)*self.W[i])+(nt*int(true_label_bits[i])*xi)
                    elif true_label_bits[i]*y_hat_ecoc_raw[i] >= 1:
                        self.W[i]=(1-nt*lmda)*self.W[i]
                    self.W[i]*=min(1, (1./math.sqrt(lmda))/np.linalg.norm(self.W[i]))

    def test(self, fname_tr, num_samples):
        confusion_matrix=np.zeros((self.num_classes, self.num_classes))
        test_labels, test_data=data_to_array(fname_tr, self.features, num_samples)

        for i in range(num_samples):
            xi=np.insert(test_data[i], 0, 1)

            y_hat_index, y_hat_ecoc = self._test_prediction(xi)
            #make prediction
            y_hat_index=int(y_hat_index)

            #set true label for readability
            true_label=test_labels[i]
            true_label_index=self.label_set.index(test_labels[i])

            if y_hat_index == true_label_index:
                confusion_matrix[y_hat_index][y_hat_index]+=1
            else:
                if y_hat_index < self.num_classes:
                    confusion_matrix[true_label_index][y_hat_index]+=1
        print('\n**** confusion matrix ****')
        print('row = true label, column = prediction')
        print(confusion_matrix, '\n')



    def prediction(self, xi):
        #prepend bias feature to xi and transpose
        xi=np.expand_dims(xi, 0)

        y_hat_ecoc=(xi@self.W.T)[0]

        y_hat_ecoc_raw=np.copy(y_hat_ecoc)

        #convert to 1s if greater than 0
        y_hat_ecoc[y_hat_ecoc >= 1] = 1
        #convert to 0 if less than or equal to 0
        y_hat_ecoc[y_hat_ecoc <= 0] = 0

        #convert ecoc to actual prediction
        y_hat_index=bin_str_to_int(y_hat_ecoc)

        return y_hat_index, y_hat_ecoc_raw

    def _test_prediction(self, xi):
        #prepend bias feature to xi and transpose
        xi=np.expand_dims(xi, 0)

        y_hat_ecoc=(xi@self.W.T)[0]

        #convert to 1s if greater than 0
        y_hat_ecoc[y_hat_ecoc > 0] = 1
        #convert to 0 if less than or equal to 0
        y_hat_ecoc[y_hat_ecoc <= 0] = 0

        #convert ecoc to actual prediction
        y_hat_index=bin_str_to_int(y_hat_ecoc)

        return y_hat_index, y_hat_ecoc
