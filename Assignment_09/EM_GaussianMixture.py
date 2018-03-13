def EM_GaussianMixture(X, NumComponents):
    import numpy as np
    from scipy.stats import multivariate_normal
    MaximumNumberOfIterations = 600
    DiffThresh = 1e-3
    N, D = X.shape
    
    # Initialize Parameters of each Component K
    Means = np.zeros((NumComponents,D))
    Sigs = np.zeros(((D, D, NumComponents)))
    Ps = np.zeros(NumComponents)
    for i in range(NumComponents):
        rVal = np.random.uniform(0,1)
        Means[i,:] = X[max(1,round(N*rVal)),:]
        Sigs[:,:,i] = 1*np.eye(D)
        Ps[i] = 1/NumComponents
        
    # E-Step Solve for p(z | x, Theta(t))
    pZ_X = np.zeros((N,NumComponents))
    for k in range(NumComponents):
        # Assign each point to a Gaussian component with probability pi(k)
        pZ_X[:,k] = multivariate_normal.pdf(X, Means[k,:], Sigs[:,:,k])*Ps[k]
    pZ_X = (pZ_X.T/np.sum(pZ_X,axis=1)).T

    Diff = np.inf
    NumberIterations = 1
    while Diff > DiffThresh and NumberIterations <= MaximumNumberOfIterations:
        # Update Means, Sigs, Ps
        MeansOld = np.array(Means)
        SigsOld = np.array(Sigs)
        PsOld = np.array(Ps)
        for k in range(NumComponents):
            #Means
            Means[k,:] = X.T@pZ_X[:,k]/sum(pZ_X[:,k])
            
            #Sigs
            xDiff = X-Means[k,:]            
            J = np.zeros((D,D))
            for i in range(N):
                J = J + pZ_X[i,k]*np.outer(xDiff[i,:], xDiff[i,:])
            Sigs[:,:,k] = J / sum(pZ_X[:,k])
            
            #Ps
            Ps[k] = sum(pZ_X[:,k]) / N
        
        # Solve for p(z | x, Theta(t))
        for k in range(NumComponents):
            # Assign each point to a Gaussian component with probability pi(k)
            pZ_X[:,k] = multivariate_normal.pdf(X, Means[k,:], Sigs[:,:,k])*Ps[k]
        pZ_X = (pZ_X.T/np.sum(pZ_X,axis=1)).T
        
        Diff = sum(sum(abs(MeansOld - Means))) + sum(sum(sum(abs(SigsOld - Sigs)))) + sum(abs(PsOld - Ps))
        print(str(NumberIterations)+" | "+str(Diff))
        NumberIterations = NumberIterations + 1
    return Means, Sigs, Ps, pZ_X
