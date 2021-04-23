import numpy as np
import cupy as cp
import math
import re
from datetime import datetime as dt
import time
from time import sleep
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

class TwoLayerNN:
    def __init__(self, features=1, num_outputs=1, labels=[1], learning_rate=.001):
        #instantiate number of inputs
        self.features=features

        #label set
        self.labels=labels

        #use this to update learning rate every x number of training samples
        self.update_rate=10000

        #instanstiate number of outputs
        print(num_outputs)
        self.num_outputs=num_outputs

        #100 hidden units for input layer
        self.h_size=100

        #hidden layer bias
        self.h_bias=1
        
        #initial learning rate
        self.learning_rate=learning_rate

        #also initial learning rate. this never changes
        self.n0=learning_rate

        #used to keep track change in validation error
        self.last_loss=0

        #used to determine when to update the learning rate
        self.loss_threshold=1

        #randomly initialize hidden layer weights
        self.h_matrix=cp.random.normal(loc=0, scale=.5, size=(self.h_size, features+1)) #+1 on features for the bias

        #set bias weight to 1 for input weight vectors
        self.h_matrix[:,0]=1

        #randomly initialize output layer weights
        self.v_matrix=cp.random.normal(loc=0, scale=.5, size=(num_outputs, self.h_size+1)) #+1 on h_size to account for bias

        #set bias weight to 1 for output weight vectors
        self.v_matrix[:,0]=1



    def train(self, fname, trng_samples, training_iterations=250):

        widgets= [ progressbar.Variable('accuracy', width=4, precision=4), ' ',
                 progressbar.Variable('wg10', width=4), ' ',
                 progressbar.Variable('wlng10', width=4), ' ',
                '[', progressbar.Timer(), ']',
                 progressbar.Bar(),
                 '(', progressbar.ETA(), ')'
        ]

        bar=progressbar.ProgressBar(max_value=training_iterations, widgets=widgets)
        prog=0

        #read in training set
        trng_labels, trng_set = data_to_array(fname, self.features, trng_samples)

        #set t0. used for learning rate update
        t0=dt.now()

        #convert to cupy arrays
        trng_labels=cp.asarray(trng_labels)
        trng_set=cp.asarray(trng_set)

        #loop for amount of training iterations
        for i in range(training_iterations):
            converge_status=False
            correct=0

            #iterate through training set
            for j in range(trng_set.shape[0]):

                #make prediction
                pred_index, prediction, vth, hn=self.trng_prediction(trng_set[j])

                #checks if prediction is incorrect
                if pred_index != self.labels.index(trng_labels[j]):
                    #calculate portion of squared loss.
                    loss=.5*(trng_labels[j]-(cp.sum(vth))**2)

                    #update loss if no change from last known loss is observed
                    #only update every thousand training examples
                    if j % self.update_rate == 0:
                        if abs(loss-self.last_loss)>=5:
                            self.last_loss=loss
                            #calculate average loss from 1000 examples
                            self.last_loss/=self.update_rate
                            t_elapsed=dt.now()-t0
                            self.learning_rate=self.n0/((t_elapsed.total_seconds()/3600)+1)

                    else:
                        self.last_loss+=loss

                    self.backpropogate(vth, hn, trng_set[j], self.labels.index(trng_labels[j]))

                else:
                    correct+=1
            bar.update(i,
                       accuracy=100.*float(correct)/trng_set.shape[0],
                       wg10=len(list(self.h_matrix[self.h_matrix > 10]) +
                                list(self.v_matrix[self.v_matrix < -10])),
                       wlng10=len(list(self.h_matrix[self.h_matrix < -10]) +
                                  list(self.v_matrix[self.v_matrix < -10])))
            correct=0




    def backpropogate(self, vth, hn, xn, y_index):
        xn=cp.pad(xn, (1,0), 'constant', constant_values=1.0)
        #set up yn
        yn=cp.zeros(vth.shape)
        yn[y_index]=1

        #set up dlda (dl with respect to activation)
        dlda=-1*(yn-vth)


        #compute dldv (dl with respect to v)
        dldv=cp.expand_dims(dlda, axis=1)*hn.T

        #step down gradient for layer 2
        self.v_matrix-=self.learning_rate*dldv

        #set up matrix for computing layer 1 derivatives
        xn_matrix=cp.tile(xn, (self.h_matrix.shape[0],1))

        f_prime=self.tanh_derivative(xn)
        f_prime=cp.pad(f_prime, ((1,0), (0,0)), 'constant', constant_values=1.0)

        #set up l1 gradient matrix
        l1_gradient_matrix=cp.zeros(self.h_matrix.shape)
        for i, vi in enumerate(self.v_matrix):
            grad=((dlda[i]*vi)*f_prime.T[0])[1:]
            grad=cp.expand_dims(grad, axis=1)*xn_matrix
            l1_gradient_matrix+=grad

        #step dow gradient for layer 1
        self.h_matrix-=l1_gradient_matrix

    
    def tanh_link(self, x):
        return cp.tanh(x)

    """
    derivative of the link function to compute first layer gradient
    tanh derivate is equal to 1-tanh(x)^2
    """
    def tanh_derivative(self, x):
        #convert to 2d array of shape (features, 1)
        x=cp.expand_dims(x, axis=1)

        #compute matrix multiplication for data with hidden layer matrix into hidden_output
        wtx=cp.matmul(self.h_matrix, x)


        #compute link function on hidden_outputs
        wtx=self.tanh_link(wtx)
        
        return cp.ones(wtx.shape)-cp.square(cp.tanh(wtx))

    """
    compute forward propogation for the neural net
    """
    def forward_propogation(self, data):
        #prepend bias feature
        data=cp.pad(data, (1,0), 'constant', constant_values=1.0)

        #convert to 2d array of shape (features, 1)
        data=cp.expand_dims(data, axis=1)

        #compute matrix multiplication for data with hidden layer matrix into hidden_output
        hidden_output=self.h_matrix@data

        #compute link function on hidden_outputs
        hidden_output=self.tanh_link(hidden_output)

        #prepend bias feature to for matrix multiplication with v
        hidden_output=cp.vstack((cp.array([cp.ones(1)]),hidden_output))

        #compute last layer output
        output=(self.v_matrix@hidden_output).T

        return output[0], hidden_output


    """
    compute answer using forward propogation
    """
    def trng_prediction(self, data):
        outputs, hidden_output=self.forward_propogation(data)
        return cp.argmax(outputs), self.labels[np.argmax(cp.asnumpy(outputs))], outputs, hidden_output

    """
    returns the index of 'outputs' that returns the highest value
    """
    def prediction(self, data):
        outputs, hidden_output=self.forward_propogation(data)
        return int(cp.argmax(outputs))

    def test(self, fname_t, num_samples, out_file):
        f=open(out_file, 'w')

        test_labels, test_data=data_to_array(fname_t, self.features, num_samples)

        #convert to cupy arrays
        test_labels=cp.asarray(test_labels)
        test_data=cp.asarray(test_data)

        confusion_matrix=np.zeros((self.num_outputs, self.num_outputs))
        np.set_printoptions(linewidth=300)
        correct=0
        for i in range(num_samples):
            y_hat=self.prediction(test_data[i])
            true_label=self.labels.index(test_labels[i])
            if y_hat == true_label:
                confusion_matrix[y_hat][y_hat]+=1
                correct+=1
            else:
                confusion_matrix[true_label][y_hat]+=1
        print('\n*** confusion matrix ***\n')
        print(confusion_matrix)
        print('accuracy:', 100.*correct/num_samples)
        f.write(str(confusion_matrix))
        f.write('\n')
        f.write('accuracy: {}'.format(100.*correct/num_samples))
        f.close()
