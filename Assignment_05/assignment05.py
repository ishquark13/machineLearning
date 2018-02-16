def assignment05():

	''' Assignment 05 for Introduction to Machine Learning, Sp 2018
		Principal Components Analysis (PCA)
	''' 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d


#Load Data G1
DS1 = np.loadtxt('G1.txt') # 10-dimensional data set (N=1500)
G1 = DS1[:,0:10]
G1_labels = DS1[:,10]

# Load Data G2
DS2 = np.loadtxt('G2.txt') # 3-dimensional data set (N=1500)
G2 = DS2[:,0:3]
G2_labels = DS2[:,3]

#Load Data G3
DS3 = np.loadtxt('G3.txt') # 3-dimensional data set (N=1500)
G3 = DS3[:,0:3]
G3_labels = DS3[:,3]

# Load Data Halfmoons
DS4 = np.loadtxt('halfmoons.txt') # 3-dimensional data set (N=1500)
moons = DS4[:,0:3]
moons_labels = DS4[:,3]

# Load Data Swissroll
swiss = np.loadtxt('swissroll.txt') # 3-dimensional data set (N=1500)
swiss_labels = np.ones(len(swiss))

# Load Data Seeds
DS6 = np.loadtxt('seeds.txt') # 7-dimensional data set (N=210)
seeds = DS6[:,0:7]
seeds_labels = DS6[:,7]

fig = plt.figure(1)
#fig.tight_layout()
# ax = fig.add_subplot(321, projection='3d')
# ax.plot3D(G1[G1_labels==1,0],G1[G1_labels==1,1], G1[G1_labels==1,2], '.r')
# ax.plot3D(G1[G1_labels==2,0],G1[G1_labels==2,1], G1[G1_labels==2,2], '.g')
# ax.plot3D(G1[G1_labels==3,0],G1[G1_labels==3,1], G1[G1_labels==3,2], '.b')
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
# ax.set_title('G1 Data Set')

# ax = fig.add_subplot(322, projection='3d')
# ax.plot3D(G2[G2_labels==1,0],G2[G2_labels==1,1], G2[G2_labels==1,2], '.r')
# ax.plot3D(G2[G2_labels==2,0],G2[G2_labels==2,1], G2[G2_labels==2,2], '.g')
# ax.plot3D(G2[G2_labels==3,0],G2[G2_labels==3,1], G2[G2_labels==3,2], '.b')
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
# ax.set_title('G2 Data Set')

# ax = fig.add_subplot(323, projection='3d')
# ax.plot3D(G3[G3_labels==1,0],G3[G3_labels==1,1], G3[G3_labels==1,2], '.r')
# ax.plot3D(G3[G3_labels==2,0],G3[G3_labels==2,1], G3[G3_labels==2,2], '.g')
# ax.plot3D(G3[G3_labels==3,0],G3[G3_labels==3,1], G3[G3_labels==3,2], '.b')
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
# ax.set_title('G3 Data Set')

# ax = fig.add_subplot(324, projection='3d')
# ax.plot3D(moons[moons_labels==1,0],moons[moons_labels==1,1], moons[moons_labels==1,2], '.r')
# ax.plot3D(moons[moons_labels==2,0],moons[moons_labels==2,1], moons[moons_labels==2,2], '.g')
# ax.plot3D(moons[moons_labels==3,0],moons[moons_labels==3,1], moons[moons_labels==3,2], '.b')
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
# ax.set_title('Halfmoons Data Set')

# ax = fig.add_subplot(325, projection='3d')
# ax.plot3D(swiss[:,0],swiss[:,1],swiss[:,2], '.b')
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
# ax.set_title('Swissroll Data Set')

# ax = fig.add_subplot(326, projection='3d')
# ax.plot3D(seeds[seeds_labels==1,3],seeds[seeds_labels==1,5], seeds[seeds_labels==1,2], '.r')
# ax.plot3D(seeds[seeds_labels==2,3],seeds[seeds_labels==2,5], seeds[seeds_labels==2,2], '.g')
# ax.plot3D(seeds[seeds_labels==3,3],seeds[seeds_labels==3,5], seeds[seeds_labels==3,2], '.b')
# ax.set_xlabel('Dimension 4')
# ax.set_ylabel('Dimension 6')
# ax.set_zlabel('Dimension 3')
# ax.set_title('Seeds Data Set')

# plt.show()

# Perform PCA


def PCA(data, corr_data, dim):
	eig_val, eig_vec = np.linalg.eig(corr_data)
	eig_pair         = [(np.abs(eig_val[i]), eig_vec[:,i]) 
	for i in range(len(eig_val))]
	eig_pair.sort(reverse = True)
	if dim == 1:
		T = np.hstack((eig_pair[0][1][:,np.newaxis]))
	else:
		T = np.hstack((eig_pair[0][1][:,np.newaxis], eig_pair[1][1][:,np.newaxis]))

	PCA_result       = data.dot(T)
	return PCA_result

def Plot_PCA_1(Data, labels):
	plt.plot(Data[labels==1,], np.zeros(np.shape(Data[labels==1, ])), '.r')
	plt.plot(Data[labels==2,], np.zeros(np.shape(Data[labels==2, ])), '.b')
	plt.plot(Data[labels==3,], np.zeros(np.shape(Data[labels==3, ])), '.g')
	plt.show()

def Plot_PCA_2(Data,labels):
	plt.plot(Data[labels==1,0], Data[labels==1,1], '.r')
	plt.plot(Data[labels==2,0], Data[labels==2,1], '.g')
	plt.plot(Data[labels==3,0], Data[labels==3,1], '.b')
	plt.show()

# 1. find means of all 6 data vectors
mean_G1    = np.mean(G1, axis = 0)
mean_G2    = np.mean(G2, axis = 0)
mean_G3    = np.mean(G3, axis = 0 )
mean_moons = np.mean(moons, axis = 0)
mean_swiss = np.mean(swiss, axis = 0)
mean_seeds = np.mean(seeds, axis = 0)

# 2. correlation of data

corr_G1    = (G1 - mean_G1).T@(G1- mean_G1)/ (G1.shape[0] - 1)
corr_G2    = (G2 - mean_G2).T@(G2- mean_G2)/ (G2.shape[0] - 1)
corr_G3    = (G3 - mean_G3).T@(G3- mean_G3)/ (G3.shape[0] - 1)
corr_moons = (moons - mean_moons).T@(moons- mean_moons)/ (moons.shape[0] - 1)
corr_swiss = (swiss - mean_swiss).T@(swiss- mean_swiss)/ (swiss.shape[0] - 1)
corr_seeds = (seeds - mean_seeds).T@(seeds- mean_seeds)/ (seeds.shape[0] - 1)

# 3. call functions for dimension reduction and plots w title

PCA_G1_dim1 = PCA(G1,corr_G1,1)
PCA_G1_dim2 = PCA(G1,corr_G1,2)
PCA_G2_dim1 = PCA(G2,corr_G2,1)
PCA_G2_dim2 = PCA(G2,corr_G2,2)
PCA_G3_dim1 = PCA(G3,corr_G3,1)
PCA_G3_dim2 = PCA(G3,corr_G3,2)
PCA_moons_dim1 = PCA(moons,corr_moons,1)
PCA_moons_dim2 = PCA(moons,corr_moons,2)
PCA_swiss_dim1 = PCA(swiss,corr_swiss,1)
PCA_swiss_dim2 = PCA(swiss,corr_swiss,2)
PCA_seeds_dim1 = PCA(seeds,corr_seeds,1)
PCA_seeds_dim2 = PCA(seeds, corr_seeds,2)

Plot_PCA_2(PCA_G1_dim2, G1_labels)
Plot_PCA_1(PCA_G1_dim1, G1_labels)
Plot_PCA_1(PCA_G2_dim1, G2_labels)
'''
Plot_PCA_2(PCA_G2_dim2, G2_labels)
Plot_PCA_1(PCA_G3_dim1, G3_labels)
Plot_PCA_2(PCA_G3_dim2, G3_labels)
Plot_PCA_1(PCA_moons_dim1, moons_labels)
Plot_PCA_2(PCA_moons_dim2, moons_labels)
Plot_PCA_1(PCA_swiss_dim1, swiss_labels)
Plot_PCA_2(PCA_swiss_dim2, swiss_labels)
Plot_PCA_1(PCA_seeds_dim1, seeds_labels)
Plot_PCA_2(PCA_seeds_dim2, seeds_labels)

'''






if __name__ == '__main__':
	assignment05()