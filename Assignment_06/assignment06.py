def assignment06():

	''' Assignment 65 for Introduction to Machine Learning, Sp 2018
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


if __name__ == '__main__':
	assignment06()
