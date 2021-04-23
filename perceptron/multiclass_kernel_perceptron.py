import numpy as np
import re
import cupy as cp
import math
from math import log
import re
from datetime import datetime as dt
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

        #convert to cupy arrays
        trng_labels=cp.asarray(trng_labels)
        trng_set=cp.asarray(trng_set)
        
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


def integer_to_bit_list(input_int, size):
    return[int(input_int) >> i&1 for i in range(size)]

class multiclass_kernel_perceptron:

    def __init__(self, features, fname_t, trng_samples, label_set, a=0, b=1, d=1):

        #a, b, and d for the kernel function (a + b*x*z)**d
        self.a=a
        self.b=b
        self.d=d

        #saves how many features a data point has
        self.features=features

        self.num_trng_samples=trng_samples

        self.label_set=label_set

        self.num_classes=len(label_set)

        #read in training set
        self.trng_labels, self.trng_set=data_to_array(fname_t, self.features, trng_samples)

        #number of classifiers
        self.num_classifiers=int(math.ceil(log(self.num_classes, 2)))

        #initialize alpha to number of training samples
        self.alpha=cp.zeros((self.num_classifiers, self.num_trng_samples))



    """
    x is the training set which is an NxD matrix

    y is the 1xD set of labels corresponding to the training set
    y is either 1 or -1

    limit is factor of 100 for multiples of training iterations.
    limit=1 means there will be 100 iterations, 2 will be 2000 and so on
    This is used because the training process doesn't converge for non-linearly
    separable data

    step size is a constant used for updating the weights

    returns: weight vector to use as icput on activation function
    """
    def learn(self, limit, step_size=1):
        widgets= [ progressbar.Variable('iteration', width=7, precision=0), ' ',
                 progressbar.Variable('accuracy'), ' ',
                '[', progressbar.Timer(), ']',
                 progressbar.Bar(),
                 '(', progressbar.ETA(), ')'
        ]

        bar=progressbar.ProgressBar(max_value=limit*100*self.trng_set.shape[0], widgets=widgets)
        prog=0

        #run through limit*100 training iterations
        for j in range(limit*100):
            count=0 #counts the number of updates
            num_correct=0
            for i in range(self.trng_set.shape[0]):
                y_hat_index, y_hat_ecoc=self.trng_prediction(self.trng_set[i])

                #take true label and separate it into array of bits
                true_label_index=self.label_set.index(self.trng_labels[i])
                true_label_bits=integer_to_bit_list(true_label_index, self.num_classifiers)

                #check if guess is wrong
                if y_hat_index != true_label_index:

                    #convert 0s in true_label_bits to -1
                    conv_true_label_bits=[1 if h > 0 else -1 for h in true_label_bits]

                    #convert 0s in y_hat_ecoc to -1
                    conv_y_hat_ecoc=[1 if h > 0 else -1 for h in y_hat_ecoc]

                    #update alpha_i for each classifier that misclassified
                    for k in range(self.num_classifiers):
                        if conv_true_label_bits[k] != conv_y_hat_ecoc[k]:
                            self.alpha[k][i]=self.alpha[k][i]+(step_size*conv_true_label_bits[k])
                    count+=1
                else:
                    num_correct+=1

                #update progress bar
                prog+=1
                if j > 0 and self.trng_set.shape[0]-i==1:
                    bar.update(prog, iteration=j, accuracy=100.*float(num_correct)/self.num_trng_samples)
                else:
                    bar.update(prog)
            if count==0:
                print("converged!")
                break
            
        if j == limit*100:
            print("did not converge")


    """
    returns index of prediction to be plugged into label_set list
    also returns y_hat_ecoc which is an array of binary digits representing y_hat_index

    only used in training
    """
    def trng_prediction(self, xi):
        xi=cp.expand_dims(xi, axis=1)

        bxz=self.b*(self.trng_set@xi).T
        kd_poly=cp.power(cp.full(bxz.shape, self.a)+bxz, self.d)

        #matrix of alphas dotted with bxz to the power d
        a=self.alpha@bxz.T
        y_hat_ecoc=cp.sum(a, axis=1)

        #convert raw predictions into 1s and 0s1
        prediction_vector=[1 if i > 0 else 0 for i in y_hat_ecoc]

        #convert ecoc to actual prediction
        y_hat_index=bit_list_to_int(prediction_vector)
        
        return y_hat_index, prediction_vector

    """
    runs through the test set and prints the confusion matrix

    fname_t is the file name of the data
    num_samples is how many samples there are
    """

    def test(self, fname_t, num_samples):
        confusion_matrix=np.zeros((self.num_classes, self.num_classes))
        
        test_labels, test_data=data_to_array(fname_t, self.features, num_samples)
        correct=0

        #iterate through test set
        for i in range(test_data.shape[0]):

            #set test data point
            xi=test_data[i]

            #make prediction
            y_hat_index, y_hat_ecoc =self.trng_prediction(xi)
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
        print('accuracy:', 100.*correct/num_samples)
        print('\n**** confusion matrix ****')
        print(confusion_matrix.astype(int))
                

