def assignment06():

	''' Assignment 6 for Introduction to Machine Learning, Sp 2018
		Classic Multidimensional Scaling (MDS), ISOMAP, Locally Liner Embedding (LLE)
	''' 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

# Load Swissroll
swissroll = np.loadtxt('swissroll.txt') # 3-dimensional data set (N=500)
swiss_labels = np.ones(len(swissroll))

# Load Clusters 
clusters = np.loadtxt('clusters.txt') # 10-dimensional data set (N=600)
clusters_labels = clusters[:,10]
clusters = clusters[:,0:10]

#Load Halfmoons
halfmoons = np.loadtxt('halfmoons.txt') 
halfmoons_labels = halfmoons[:,3] 
halfmoons = halfmoons[:,0:3]

fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(131, projection='3d')
ax.plot3D(swissroll[:,0],swissroll[:,1], swissroll[:,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Swissroll Data Set')

ax = fig.add_subplot(132, projection='3d')
ax.plot3D(clusters[clusters_labels==1,0],clusters[clusters_labels==1,1], clusters[clusters_labels==1,2], '.r')
ax.plot3D(clusters[clusters_labels==2,0],clusters[clusters_labels==2,1], clusters[clusters_labels==2,2], '.g')
ax.plot3D(clusters[clusters_labels==3,0],clusters[clusters_labels==3,1], clusters[clusters_labels==3,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Clusters Data Set')

ax = fig.add_subplot(133, projection='3d')
ax.plot3D(halfmoons[halfmoons_labels==1,0],halfmoons[halfmoons_labels==1,1], halfmoons[halfmoons_labels==1,2], '.r')
ax.plot3D(halfmoons[halfmoons_labels==2,0],halfmoons[halfmoons_labels==2,1], halfmoons[halfmoons_labels==2,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Halfmoons Data Set')

plt.show()

# Implement Classic MDS, ISOMAP and LLE
def Plot_PCA_2(Data,labels):
	plt.plot(Data[labels==1,0], Data[labels==1,1], '.r')
	plt.plot(Data[labels==2,0], Data[labels==2,1], '.g')
	plt.plot(Data[labels==3,0], Data[labels==3,1], '.b')
	plt.show()

def MDS(data):
	#1. Parse data
	#2. Compute proximity matrix D
	#3. Eigendecomp
	#4. y = Vlambda1/2
    
    #@params: takes in a distance matrix

	#D 	= squareform(pdist(data,metric='euclidean'))
	D_2 = np.square(data)
	print(np.shape(D_2)) #500x500
	N   = data.shape[0]
	J   = np.identity(N)-1/N*np.ones((N, 1))@np.ones((N, 1)).T
	B   = -0.5 * J.dot(D_2).dot(J)

	evals,evecs = np.linalg.eigh(B)

	new_dim = np.argsort(evals)[::-1]
	evals   = evals[new_dim]
	evecs   = evecs[:,new_dim]

	#diag, = np.where(evals > 0)
	Lambda = np.diag(np.sqrt(evals[0:2]))
	V      = evecs[:,0:2]
	output = V.dot(Lambda)
	return output

def ISOMAP(data, K):
    #k = 100
    nbr                = NearestNeighbors(K+1, metric = 'euclidean').fit(data)
    distances, indices = nbr.kneighbors(data)
    Dx                 = squareform(pdist(data, metric =  'euclidean'))
    graph              = nbr.kneighbors_graph(data,K+1).toarray()
    adjacency_matrix   = np.multiply(graph,Dx)
    print(adjacency_matrix)
    return adjacency_matrix
    

def floyd_mayweather_algorithm(m):
    
    m[m == float(0)] = 10**10
    m = m - np.eye(len(m[0]))*10**10
    '''
    for k in range(len(m[0])):
        for i in range(len(m[0])):
            for j in range(len(m[0])):
                m[i,j] = min(m[i,j], m[i,k] + m[k,j])
    '''            
    
    for k in range(len(m)):
        m = np.minimum(m, m[np.newaxis,k,:] + m[:,k,np.newaxis])             
    return m            
       

if __name__ == '__main__':
	assignment06() 
'''
	johns = MDS(swissroll)
	print('A')
	print(np.shape(johns))
	print('A')
	Plot_PCA_2(johns, swiss_labels)
'''
# for use with MDS    
'''
A 	= squareform(pdist(swissroll,metric='euclidean'))
B 	= squareform(pdist(halfmoons,metric='euclidean'))
D 	= squareform(pdist(clusters,metric='euclidean'))
mds = MDS(D)
'''

isomap_matrix = ISOMAP(clusters, 15)
johns = floyd_mayweather_algorithm(isomap_matrix)
johns2 = MDS(johns)

Plot_PCA_2(johns2,clusters_labels)

    
    