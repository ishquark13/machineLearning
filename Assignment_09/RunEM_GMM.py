from EM_GaussianMixture import EM_GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def RunEM_GMM(data,labels,NumComponents,name):

    Means, Sigs, Ps, pZ_X = EM_GaussianMixture(data,NumComponents)
    
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(2,NumComponents,1, projection='3d')
    p1 = ax.scatter(data[:,0], data[:,1], data[:,2], c=labels) 
    ax.set_title('Data in ' + name + ' Color Model')
    fig.colorbar(p1, ax=ax)
    for i in range(NumComponents):
        ax = fig.add_subplot(2,NumComponents,i+1+NumComponents, projection='3d')
        p1 = ax.scatter(data[:,0], data[:,1], data[:,2], c=pZ_X[:,i]) 
        ax.set_title('Mean: '+ str(Means[i,:]))
        fig.colorbar(p1, ax=ax)
    plt.suptitle('EM w/ GMM in ' + name + ' Color Model')
    
    return Means, Sigs, Ps, pZ_X