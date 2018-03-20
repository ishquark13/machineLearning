from pcm import pcm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def RunPCM(data,labels,NumClusters,flag,name):


    centers, U = pcm(data, NumClusters, m=2, eta=0.5)         

    #U = U + 1
'''
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    p1 = ax.scatter(data[:,0], data[:,1],data[:,2], c=labels) 
    ax.set_title('Data in ' + name +' Color Model')
    fig.colorbar(p1, ax=ax)
    ax = fig.add_subplot(122, projection='3d')
    p1 = ax.scatter(data[:,0],data[:,1],data[:,2],c=U)
    ax.set_title('PCM in ' + name + ' Color Model')
    fig.colorbar(p1, ax=ax)
    plt.suptitle('PCM of Data in ' + name + ' Color Model')
'''
    return centers, U