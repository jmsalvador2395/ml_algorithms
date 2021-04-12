import numpy as np
import cupy as cp
import math
import re
from datetime import datetime as dt

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
        def __init__(self, features=1, num_outputs=1, labels=[1], learning_rate=1):
                #instantiate number of inputs
                self.features=features

                #label set
                self.labels=labels

                #use this to update learning rate every x number of training samples
                self.update_rate=10000

                #instanstiate number of outputs
                self.num_outputs=num_outputs

                #100 hidden units for input layer
                self.h_size=100

                #hidden layer bias
                self.h_bias=1
                
                #initial learning rate. subtract 5 when error stops decreasing
                self.learning_rate=learning_rate

                #also initial learning rate. this never changes
                self.n0=learning_rate

                #used to keep track change in validation error
                self.last_loss=0

                #used to determine when to update the learning rate
                self.loss_threshold=1

                #randomly initialize hidden layer weights
                self.h_matrix=cp.random.normal(loc=0, scale=1, size=(self.h_size, features+1)) #+1 on features for the bias

                #set bias weight to 1 for input weight vectors
                self.h_matrix[:,0]=1

                #randomly initialize output layer weights
                self.v_matrix=cp.random.normal(loc=0, scale=1, size=(num_outputs, self.h_size+1)) #+1 on h_size to account for bias

                #set bias weight to 1 for output weight vectors
                self.v_matrix[:,0]=1



        def train(self, fname, trng_samples, training_iterations=500):

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

                    #iterate through training set
                    for j in range(trng_set.shape[0]):

                        #make prediction
                        pred_index, prediction, vth, hn=self.trng_prediction(trng_set[j])

                        #checks if prediction is incorrect
                        if prediction != trng_labels[j]:
                            #calculate portion of squared loss.
                            loss=.5*((trng_labels[j]-cp.sum(vth))**2)

                            #update loss if no change from last known loss is observed
                            #only update every thousand training examples
                            if j % self.update_rate == 0:
                                if abs(loss-self.last_loss)>=5:
                                    self.last_loss=loss
                                    #calculate average loss from 1000 examples
                                    self.last_loss/=self.update_rate
                                    t_elapsed=dt.now()-t0
                                    self.learning_rate=self.n0/(5*t_elapsed.total_seconds()/60.)

                            else:
                                self.last_loss+=loss

                            #layer updated depending on training iteration
                            if i%2 == 1:
                                #update layer 1 weights with gradient descent
                                l1_gradient=self.layer_1_gradient_matrix(trng_labels[j], vth, trng_set[j], pred_index, hn)
                                #print('l1 {} {} {} {}\n'.format(self.learning_rate, loss, j, i),self.h_matrix,'\n')
                                self.h_matrix-=l1_gradient*self.learning_rate

                            else:
                                #update layer 2 weights with gradient descent
                                l2_gradient=self.layer_2_gradient_matrix(trng_labels[j], vth, pred_index, hn)
                                self.v_matrix-=l2_gradient
                                print('l2 {} {} {} {} {}\n'.format(self.learning_rate, loss, j, i, prediction),self.v_matrix,'\n', l2_gradient)

                    

        """
        first layer squared loss gradient
        gradient taken with respect to w
        """
        def layer_1_gradient_matrix(self, yn, vth, xn, vi_index, hn):
            #initialize gradient matrix
            l1_gradient_matrix=cp.zeros(self.h_matrix.shape)

            #prepend bias feature to xn
            xn=cp.pad(xn, (1,0), 'constant', constant_values=1.0)

            #compute derivative of link function
            f_prime=self.tanh_derivative(xn)*cp.expand_dims(xn, axis=0)

            #create yn for each v vector
            yn=cp.ones(vth.shape)*-1
            yn[vi_index]=1

            #sum up l1_gradient matrices for each vi
            for i in range(self.num_outputs):
                vi=self.v_matrix[i]
                l1_gradient_matrix+=(f_prime*cp.expand_dims(xn, axis=0))*(-1*(yn[i]-cp.dot(vi,hn)))

            return l1_gradient_matrix

        """
        second layer squared loss gradient
        gradient taken with respect to v
        """
        def layer_2_gradient_matrix(self, yn, vth, vi_index, hn):
            #reshape vth to be compatible with matrix multiplications
            vth=cp.expand_dims(vth, axis=1)

            #compute yn
            yn=cp.zeros(vth.shape)
            yn[0][vi_index]=1

            #print(yn-vth)
            #print(vth, "\n")
            #print(cp.matmul((yn-vth),hn.T)*-1, '\n')
            return -1*cp.matmul((yn-vth),hn.T)

        
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
            #return cp.ones(x.shape)-cp.square(cp.tanh(x))

        """
        compute forward propogation for the neural net
        """
        def forward_propogation(self, data):
                #prepend bias feature
                data=cp.pad(data, (1,0), 'constant', constant_values=1.0)

                #convert to 2d array of shape (features, 1)
                data=cp.expand_dims(data, axis=1)

                #compute matrix multiplication for data with hidden layer matrix into hidden_output
                hidden_output=cp.matmul(self.h_matrix, data)

                #compute link function on hidden_outputs
                hidden_output=self.tanh_link(hidden_output)

                #prepend bias feature to for matrix multiplication with v
                hidden_output=cp.vstack((cp.array([cp.ones(1)]),hidden_output))

                #compute last layer output
                output=cp.matmul(hidden_output.T, self.v_matrix.T)


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
            return self.labels[cp.argmax(outputs)]
    
