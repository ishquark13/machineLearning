import numpy as np
from scipy.spatial.distance import cdist
from numpy.random import permutation
from numpy.linalg import norm

def pcm(X, M, m, eta):
    #X is the input data, Each row is a data point
    #M is the number of clusters
    #m is the fuzzifier
    #eta is the weights in the second term of the objective function (could be scalar of vector of size Nx1)
    
    #Parameters
    MaxIter = 1000
    StopThresh = 1e-5
    
    #Initialize Cluster Centers by drawing Randomly from Data (can use other methods for initialization...)
    N, d = np.shape(X) # number of data points, dimensionality
        
    eta = eta*np.ones(M) # Weights for every points
    # Considering it fixed and equal to all points. But they could be different or even updated as the clustering evolves
    
    rp = permutation(N) #random permutation of numbers 1:N
    centers = X[rp[0:M],:] #select first M data points sorted according to rp
    
    D = cdist(X, centers, metric = 'euclidean')
    
    #eta = .001*np.ones(N)
    
    U = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            U[i,j] = 1/(1+ ((D[i,j]/eta[j])**(1/(m-1))))
        U[i,:] = U[i,:]/sum(U[i,:])
    
    diff = np.inf
    it = 0
    while diff > StopThresh and it < MaxIter:
        # Update cluster centers
        centersPrev = centers
        for i in range(M):
            num = np.zeros(d)
            den = np.zeros(d)
            for j in range(N):
                num = num + U[j,i]**m * centersPrev[i,:]
                den = den + U[j,i]**m
            centers[i,:] = np.squeeze(num/den)
        
        # Update distance to centers
        D = cdist(X, centers, metric='euclidean')**2
        
        # Update U
        UPrev = U
        for i in range(N):
            for j in range(M):
                U[i,j] = 1/(1+ ((D[i,j]/eta[j])**(1/(m-1))))
            U[i,:] = U[i,:]/sum(U[i,:])

        # Update diff & iteration count for stopping criteria
        diff = norm(UPrev - U)
        it+=1

    return centers, U
