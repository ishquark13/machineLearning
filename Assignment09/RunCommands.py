	
''' Assignment 08 for Introduction to Machine Learning, Sp 2018
		Clustering: K-Means, Fuzzy C-Means & Possibilistic C-Means
''' 

from kmeans import KMeans
from fcm import cmeans
from pcm import pcm
import numpy as np
import matplotlib.pyplot as plt



def RunKMeans(data, labels, n_clusters, name, display):
    kM_centers, kM_labels = KMeans(data, n_clusters, display)
    fig = plt.figure(figsize=(18,8))
    fig.add_subplot(1,2,1)
    plt.scatter(data[:,0],data[:,1],c=labels)
    plt.title(name)
    fig.add_subplot(1,2,2)
    plt.scatter(data[:, 0], data[:, 1], c=kM_labels)
    plt.title("K-Means Clustering")
    # plt.show()
    return kM_centers, kM_labels


    
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
    # plt.show()

    return c_FCM, L_FCM
        
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
    # plt.show()
    return c_PCM, L_PCM

