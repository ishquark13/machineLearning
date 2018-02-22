	
''' Assignment 08 for Introduction to Machine Learning, Sp 2018
		Clustering: K-Means, Fuzzy C-Means & Possibilistic C-Means
''' 

from kmeans import KMeans
from fcm import cmeans
from pcm import pcm
import numpy as np
import matplotlib.pyplot as plt

# Load Data Sets
X1 = np.loadtxt('X1.txt')
labels1 = np.loadtxt('labels1.txt')

X2 = np.loadtxt('X2.txt')
labels2 = np.loadtxt('labels2.txt')

X3 = np.loadtxt('X3.txt')
labels3 = np.loadtxt('labels3.txt')

X4 = np.loadtxt('X4.txt')
labels4 = np.loadtxt('labels4.txt')

X5 = np.loadtxt('X5.txt')
labels5 = np.loadtxt('labels5.txt')

X6 = np.loadtxt('X6.txt')
labels6 = np.loadtxt('labels6.txt')

fig = plt.figure(figsize=(18,8))
plt.suptitle('Data Sets')
fig.add_subplot(2,3,1)
plt.scatter(X1[:,0],X1[:,1],c=labels1)
plt.title('Blobs (3 clusters)')
fig.add_subplot(2,3,2)
plt.scatter(X2[:,0],X2[:,1],c=labels2)
plt.title('Blobs with different variances (3 clusters)')
fig.add_subplot(2,3,3)
plt.scatter(X3[:,0],X3[:,1],c=labels3)
plt.title('Half Moons (2 clusters)')
fig.add_subplot(2,3,4)
plt.scatter(X4[:,0],X4[:,1],c=labels4)
plt.title('Anisotropicly Blobs (3 clusters)')
fig.add_subplot(2,3,5)
plt.scatter(X5[:,0],X5[:,1],c=labels5)
plt.title('Circles (2 clusters)')
fig.add_subplot(2,3,6)
plt.scatter(X6[:,0],X6[:,1],c=labels6)
plt.title('No Structure (1 custer)')

def RunKMeans(data, labels, n_clusters, name, display):
    kM_centers, kM_labels = KMeans(data, n_clusters, display)
    fig = plt.figure(figsize=(18,8))
    fig.add_subplot(1,2,1)
    plt.scatter(data[:,0],data[:,1],c=labels)
    plt.title(name)
    fig.add_subplot(1,2,2)
    plt.scatter(data[:, 0], data[:, 1], c=kM_labels)
    plt.title("K-Means Clustering")
    
def RunFCM(data, labels, n_clusters, m, name):
    c_FCM, L_FCM = cmeans(data.T, n_clusters, m)
    fig = plt.figure(figsize=(18,8))
    fig.add_subplot(2,n_clusters,1)
    plt.scatter(data[:,0],data[:,1],c=labels)
    plt.title(name)
    for i in range(n_clusters):
        ax = fig.add_subplot(2,n_clusters,n_clusters+i+1)
        p1 = plt.scatter(data[:, 0], data[:, 1], c=L_FCM[i,:])
        plt.title("Fuzzy C-Means Clustering")
        fig.colorbar(p1, ax=ax)
        
def RunPCM(data, labels, n_clusters, m, eta, name):
    c_PCM, L_PCM = pcm(data, n_clusters, m, eta)
    fig = plt.figure(figsize=(18,8))
    fig.add_subplot(2,n_clusters,1)
    plt.scatter(data[:,0],data[:,1],c=labels)
    plt.title(name)
    for i in range(n_clusters):
        ax = fig.add_subplot(2,n_clusters,n_clusters+i+1)
        p1 = plt.scatter(data[:, 0], data[:, 1], c=L_PCM[:,i])
        plt.title("Possibilistic C-Means Clustering")
        fig.colorbar(p1, ax=ax)

#%% Run K-Means

RunKMeans(data=X1, labels=labels1, n_clusters=3, name="Data Set X1", display=1)


#%% Run Fuzzy C-Means (FCM)

RunFCM(data=X4, labels=labels4, n_clusters=5, m=2, name="Data Set X4")


#%% Run Possibilistic C-Means (PCM)

RunPCM(data=X6, labels=labels6, n_clusters=2, m=1.1, eta=0.5, name="Data Set X6")

