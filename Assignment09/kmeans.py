def KMeans(X, C, flag):
    import numpy as np
    from scipy import spatial
    import matplotlib.pyplot as plt
    MaxIter     = 10000
    StopThresh  = 1e-5
    
    #Initialize Cluster Centers by drawing Randomly from Data (can use other
    # methods for initialization...)
    N       = X.shape[0] #number of data points
    rp      = np.random.permutation(N) #random permutation of numbers 1:N
    centers = X[rp[0:C],:] #select first M data points sorted according to rp
    
    diff    = 1e100
    iter    = 0
    if flag:
        plt.figure()
    colors = ['c','m','b','r','g','c','m','y']
    while((diff > StopThresh) and (iter < MaxIter)):
    #Assign data to closest cluster representative (using Euclidean distance)                
        D   = spatial.distance.cdist(X, centers)
        L   = np.argmin(D, axis=1)
        
        if flag:
            if iter==0:
                plt.scatter(X[:,0],X[:,1],c='k')
                for j in range(C):
                    plt.scatter(centers[j,0],centers[j,1],c=colors[j],marker='x', s=200)
                    # plt.show()
                plt.title("K-Means - Iteration: "+str(iter))
                plt.pause(0.5) 
            else:
                plt.cla()
                for j in range(C):
                    plt.scatter(X[L==j,0],X[L==j,1],c=colors[j])
                    plt.scatter(centers[j,0],centers[j,1],c='k',marker='x', s=200)
                    # plt.show()
                plt.title("K-Means - Iteration: "+str(iter))
                plt.pause(0.5)
        	    
        #Update cluster centers
        centersPrev = centers.copy()
        for i in range(C):
            centers[i,:] = np.mean(X[L == i,:], axis=0)
        
        #Update diff & iteration count for stopping criteria
        diff = np.linalg.norm(centersPrev - centers)
        iter = iter+1
    return centers, L
