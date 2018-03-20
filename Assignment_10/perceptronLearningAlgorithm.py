import numpy as np
import matplotlib.pyplot as plt

def generateMVNRandData(Npts, mu, sigma):
	data  = np.random.multivariate_normal(mu, sigma*np.eye(len(mu)), Npts)
	return data

def plotLine(weights, range):
	x = np.array(range)
	y = -(weights[0]/weights[1])-(weights[2]/weights[1])*x
	plt.plot(y,x)
	plt.pause(2)

def perceptronLearningAlg(data,labels,eta,nIterations):
    nPts = data.shape[0]
    weights = np.random.rand(data.shape[1])
    print('Initial weights:', weights)
    
    error = 1
    iter = 0
    while(error > 0 & iter < nIterations):
        print('Iteration: ', iter,'; Error: ', error)
        error = 0
        iter += 1
        for i in range(nPts):
            activation =  data[i,:]@weights
            activation = (activation>0)
            if (activation-labels[i])!=0:
                plt.cla()
                weights-=eta*data[i,:]*(activation-labels[i])
                error += 1
                plt.scatter(data[:,1],data[:,2], c=labels, linewidth=0)
                plotLine(weights, [-2,2])
            
    print('Final Iteration: ', iter,'; Final Error: ', error)
    return weights

#%%
if __name__ == '__main__':
	Npts  = 100
	mu1   = [2,2]
	mu2   = [0,0]
	var   = .1
	eta   = 10
	nIterations = 10;

	plt.ion()
	fig   = plt.figure()

	data1 = np.array(generateMVNRandData(Npts, mu1, .1))
	data1 = np.hstack((np.ones((Npts,1)),data1))
	data2 = np.array(generateMVNRandData(Npts, mu2, .1))
	data2 = np.hstack((np.ones((Npts,1)),data2))
	data  = np.vstack(( data1, data2))
	labels= np.hstack((np.ones(Npts), np.zeros(Npts)))

	plt.scatter(data[:,1],data[:,2], c=labels, linewidth=0)
	plt.pause(2)

	perceptronLearningAlg(data,labels,eta,nIterations)