from kmeans import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def RunKMeans(data,labels,NumClusters,flag,name):
    
    centers, L = KMeans(data, NumClusters, flag)
    
    L = L+1
    
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    p1 = ax.scatter(data[:,0], data[:,1],data[:,2], c=labels) 
    ax.set_title('Data in ' + name +' Color Model')
    fig.colorbar(p1, ax=ax)
    ax = fig.add_subplot(122, projection='3d')
    p1 = ax.scatter(data[:,0],data[:,1],data[:,2],c=L)
    ax.set_title('K-Means in ' + name + ' Color Model')
    fig.colorbar(p1, ax=ax)
    plt.suptitle('K-Means of Data in ' + name + ' Color Model')
    
    return centers, L