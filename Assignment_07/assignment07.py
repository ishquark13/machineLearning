def assignment07():

	''' Assignment 07 for Introduction to Machine Learning, Sp 2018
		EM Algorithm and Gaussian Mixture Models
	''' 
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Load Data
X = np.loadtxt('GaussianMixture.txt')
fig = plt.figure()
plt.plot(X[:,0],X[:,1],'.b')
NumComponents = 3

def EM_GaussianMixture(X, NumComponents):
    MaximumNumberOfIterations = 600
    DiffThresh = 1e-4
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
        # C_ik = pZ_X in bayesian terms
        pZ_X[:,k] = multivariate_normal.pdf(X,Means[k,:], Sigs[:,:,k])*Ps[k]
        #normalize the rows to add up to 1
    for i in range(N):
        pZ_X[i,:] /= np.sum(pZ_X[i,:])
        
    Diff = np.inf
    NumberIterations = 1
     
    #init the covariance matrix for cluster k
    #sigma_cluster  = np.zeros((D,D))
    
    while Diff > DiffThresh and NumberIterations <= MaximumNumberOfIterations:
            ## M-step: Update Means, Sigs, Ps
            MeansOld = np.array(Means)
            SigsOld = np.array(Sigs)
            PsOld = np.array(Ps)
            #vectorized format
            Means = ((X.T@pZ_X)/(np.ones((D,1))*np.sum(pZ_X, axis = 0))).T
            Ps = np.sum(pZ_X, axis = 0)/N
             
            
            
            for k in range(NumComponents):
               # Complete M-step: Update parameters
               #Means[k,:] = (np.sum(pZ_X[:,k]*X.T, axis = 1).T)/(np.sum(pZ_X[:,k])) 
               X_mid = X - Means[k,:]
               J  = np.zeros((D,D))
               for i in range(N):
                   J  = J + pZ_X[i,k]*np.outer(X_mid[i,:],X_mid[i,:])
               Sigs[:,:,k]        = J/np.sum(pZ_X[:,k])
               # pZ_Xindex = np.sum(pZ_X, axis = 0)
               # Sigs =  Sigs[k] = np.array(1 / pZ_Xindex * np.dot(np.multiply(x_mu.T,  pZ_X[:, k]), x_mu))
               ## E-step: Solve for p(z | x, Theta(t))
               # Complete E-step
               #Ps[k] = np.sum(pZ_X[:,k])/N
               
               
               #Reiterate same as above
            for k in range(NumComponents):
               # Assign each point to a Gaussian component with probability pi(k)
               # C_ik
                   
                   pZ_X[:,k] = multivariate_normal.pdf(X,Means[k,:], Sigs[:,:,k])*Ps[k]
            for i in range(N):
                   pZ_X[i,:] /= np.sum(pZ_X[i,:])
                   
                   
            Diff = sum(sum(abs(MeansOld - Means))) + sum(sum(sum(abs(SigsOld - Sigs)))) + sum(abs(PsOld - Ps))
            print(NumberIterations)
            NumberIterations = NumberIterations + 1
    return Means, Sigs, Ps, pZ_X

# Set number of componenets
NumComponents = 3
EM_Means, EM_Sigs, EM_Ps, pZ_X = EM_GaussianMixture(X, NumComponents)

print('----------------------------')
print('----------------------------')
print('EM Algorithm')
print('# Components: ' + str(NumComponents))
print('----------------------------')
print('Estimated Means')
print(EM_Means)
print('Estimated Covariances')
for i in range(NumComponents):
    print(EM_Sigs[:,:,i])
    print('Estimated Weights')
    print(EM_Ps)
    
fig = plt.figure(figsize=(15, 4))
plt.suptitle('EM Algorithm')
for i in range(NumComponents):
    ax = fig.add_subplot(1,NumComponents,i+1)
    p1 = ax.scatter(X[:,0], X[:,1], c=pZ_X[:,i]) 
    ax.set_title('Mean: '+ str(EM_Means[i,:]))
    fig.colorbar(p1, ax=ax)

if __name__ == '__main__':
    assignment07()

