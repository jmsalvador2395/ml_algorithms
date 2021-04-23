import numpy as np
import cupy as cp
import re
import math
from math import log
import re
from datetime import datetime as dt
import random
import time
import progressbar
 
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
converts list of binary digits into decimal number
"""
def bit_list_to_int(inpt):
    integer=0
    for i,bit in enumerate(inpt):
        if bit == 1:
            integer+=2**i
    return integer



"""
converts integer into list of 0s and 1s
index 0 is the least significant bit
"""
def integer_to_bit_list(input_int, size):
    return[int(input_int) >> i&1 for i in range(size)]

"""
returns a or b depending on which is smaller
"""
def min_ab(a, b):
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

        self.num_trng_samples=self.trng_labels.size

        #number of classes
        self.num_classes=len(label_set)

        #number of classifiers
        self.num_classifiers=int(math.ceil(log(self.num_classes, 2)))

        #initialize weight vectors
        self.W=np.zeros((self.num_classifiers, features+1))

        self.t0=dt.now()


    """
    implemented for numpy arrays

    trng_set is a TxD numpy array
    labels is a 1d numpy array
    lmda is the scalar
    T is the amount of samples
    """
    def pegasos(self, lmda, T):
        widgets= [ progressbar.Variable('accuracy', width=4, precision=4), ' ',
                 progressbar.Variable('wge0', width=4), ' ',
                 progressbar.Variable('wl0', width=4), ' ',
                 progressbar.Variable('w1neg1', width=4), ' ',
                '[', progressbar.Timer(), ']',
                 progressbar.Bar(),
                 '(', progressbar.ETA(), ')'
        ]

        T=T*self.trng_set.shape[0]

        bar=progressbar.ProgressBar(max_value=T, widgets=widgets)
        prog=0
        w=np.zeros(self.features+1)

        random.seed()
        interval=T/10000

        correct=0
        for t in range(T):
            it=np.random.randint(0,self.trng_set.shape[0]-1)
            nt=1./(lmda*(t+1)) #calculate step size

            #prepend bias feature to xi and transpose
            xi=np.insert(self.trng_set[it], 0, 1)

            #make prediction
            y_hat_index, y_hat_ecoc_raw=self.prediction(xi)

            #take true label and separate it into array of bits
            true_label_index=self.label_set.index(self.trng_labels[it])
            true_label_bits=integer_to_bit_list(true_label_index, self.num_classifiers)

            #convert 0s in true_label_bits to -1
            conv_true_label_bits=[i if i > 0 else -1 for i in true_label_bits]

            if y_hat_index == true_label_index:
                correct+=1

            for i in range(self.num_classifiers):
                label=conv_true_label_bits[i]
                w=self.W[i]
                step_size=(1-(nt*lmda))
                if label*y_hat_ecoc_raw[i] < 1:
                    self.W[i]=(step_size*w)+(nt*label*xi)

                elif label*y_hat_ecoc_raw[i] >= 1:
                    self.W[i]=step_size*w

                w_norm=np.linalg.norm(self.W[i])
                if w_norm != 0:
                    self.W[i]=min_ab(1, ((1./math.sqrt(lmda))/w_norm))*self.W[i]
            prog+=1
            if t%interval == 0:
                bar.update(prog,
                           accuracy=100.*float(correct)/interval,
                           wge0=len(self.W[self.W >= 0]),
                           wl0=len(self.W[self.W < 0]),
                           w1neg1=len(self.W[(self.W<1)*(self.W>-1)])) 
                correct=0
            else:
                bar.update(prog)


    def test(self, fname_tr, num_samples):
        confusion_matrix=np.zeros((self.num_classes, self.num_classes))
        test_labels, test_data=data_to_array(fname_tr, self.features, num_samples)

        correct=0
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
                correct+=1
            else:
                if y_hat_index < self.num_classes:
                    confusion_matrix[true_label_index][y_hat_index]+=1
        print('\n**** confusion matrix ****')
        print('row = true label, column = prediction')
        print(confusion_matrix, '\n')

        print('\naccuracy: {}'.format(100.*correct/num_samples))



    def prediction(self, xi):
        #prepend bias feature to xi and transpose
        xi=np.expand_dims(xi, 1)

        y_hat_ecoc=(self.W@xi).T[0]

        y_hat_ecoc_raw=np.copy(y_hat_ecoc)


        #convert to 1s if greater than or equal to 1
        y_hat_ecoc[y_hat_ecoc > 0] = 1
        #convert to 0 if less than 1
        y_hat_ecoc[y_hat_ecoc <= 0] = 0
        
        #convert ecoc to actual prediction
        y_hat_index=bit_list_to_int(y_hat_ecoc)

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
        y_hat_index=bit_list_to_int(y_hat_ecoc)

        return y_hat_index, y_hat_ecoc
