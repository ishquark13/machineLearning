
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
    
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            print(count)
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        print("Stopped", new_val_error,old_val_error1, old_val_error2)
        return new_val_error
    	
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        fig   = plt.figure()
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
            
        error = np.zeros(niterations)
        for n in range(niterations):
    
            self.outputs = self.mlpfwd(inputs)

            error[n] = 0.5*np.sum((self.outputs-targets)**2)
            if (np.mod(n,100)==0):
                print("Iteration: ",n, " Error: ",error[n])  
                #plt.scatter(n,error[n])

            # Different types of output neurons
            if self.outtype == 'linear':
            	deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'logistic':
            	deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata 
            else:
            	print("error")
            
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
                      
            updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2
                
            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
        plt.plot(error,'-o')
        plt.xlabel('Iteration n')
        plt.ylabel('Cost Function J(n)')
        plt.title('Learning Curve')
        plt.legend(["Learning rate = "+ str(eta)])
        plt.plot(np.arange(0,niterations+1),np.zeros(niterations+1),'--r')
        plt.show()
        return self.weights1, self.weights2
    
    def mlptrain_online(self,inputs,targets,eta,niterations):
        """ Train the thing online """    
        # Add the inputs that match the bias node
        fig   = plt.figure()
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = np.arange(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
            
        error = np.zeros(niterations)
        for n in range(niterations):
            
            np.random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]
            self.outputs = self.mlpfwd(inputs)

            error[n] = 0.5*np.sum((self.outputs-targets)**2)

            if (np.mod(n,100)==0):
                print("Iteration: ",n, " Error: ",error[n])  
                #plt.scatter(n,error[n])

            for i in range(self.ndata):

                # Different types of output neurons
                if self.outtype == 'linear':
                    deltao = (self.outputs[i,:] - targets[i,:])
                elif self.outtype == 'logistic':
                    deltao = self.beta * (self.outputs[i,:] - targets[i,:]) * self.outputs[i,:] * (1.0 - self.outputs[i,:])
                elif self.outtype == 'softmax':
                    deltao = (self.outputs[i,:] - targets[i,:]) * (self.outputs[i,:] * (-self.outputs[i,:]) + self.outputs[i,:]) 
                else:
                    print("error")
                
                deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
                deltah_reshape = np.reshape(deltah[i,:-1],(2,1))

                sample = np.reshape(inputs[i,:],(5,1))
                hidden_reshape = np.reshape(self.hidden[i,:],(3,1))
                deltao_reshape = np.transpose(np.reshape(deltao,(3,1)))
                          
                updatew1 = eta*(np.dot((sample),(np.transpose(deltah_reshape)))) + self.momentum*updatew1
                updatew2 = eta*(np.dot((hidden_reshape),(deltao_reshape))) + self.momentum*updatew2
                self.weights1 -= updatew1
                self.weights2 -= updatew2
                    
                # Randomise order of inputs (not necessary for matrix-based calculation)
                #np.random.shuffle(change)
                #inputs = inputs[change,:]
                #targets = targets[change,:]

        plt.plot(error,'-o')
        plt.xlabel('Iteration n')
        plt.ylabel('Cost Function J(n)')
        plt.title('Learning Curve')
        plt.legend(["Learning rate = "+ str(eta)])
        plt.plot(np.arange(0,niterations+1),np.zeros(niterations+1),'--r')
        plt.show()
        return self.weights1, self.weights2

    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print("error")

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)
