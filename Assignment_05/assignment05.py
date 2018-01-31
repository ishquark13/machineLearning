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
halfmoons = DS4[:,0:3]
halfmoons_labels = DS4[:,3]

# Load Data Swissroll
swissroll = np.loadtxt('swissroll.txt') # 3-dimensional data set (N=1500)

# Load Data Seeds
DS6 = np.loadtxt('seeds.txt') # 7-dimensional data set (N=210)
seeds = DS6[:,0:7]
seeds_labels = DS6[:,7]

fig = plt.figure(1)
#fig.tight_layout()
ax = fig.add_subplot(321, projection='3d')
ax.plot3D(G1[G1_labels==1,0],G1[G1_labels==1,1], G1[G1_labels==1,2], '.r')
ax.plot3D(G1[G1_labels==2,0],G1[G1_labels==2,1], G1[G1_labels==2,2], '.g')
ax.plot3D(G1[G1_labels==3,0],G1[G1_labels==3,1], G1[G1_labels==3,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('G1 Data Set')

ax = fig.add_subplot(322, projection='3d')
ax.plot3D(G2[G2_labels==1,0],G2[G2_labels==1,1], G2[G2_labels==1,2], '.r')
ax.plot3D(G2[G2_labels==2,0],G2[G2_labels==2,1], G2[G2_labels==2,2], '.g')
ax.plot3D(G2[G2_labels==3,0],G2[G2_labels==3,1], G2[G2_labels==3,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('G2 Data Set')

ax = fig.add_subplot(323, projection='3d')
ax.plot3D(G3[G3_labels==1,0],G3[G3_labels==1,1], G3[G3_labels==1,2], '.r')
ax.plot3D(G3[G3_labels==2,0],G3[G3_labels==2,1], G3[G3_labels==2,2], '.g')
ax.plot3D(G3[G3_labels==3,0],G3[G3_labels==3,1], G3[G3_labels==3,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('G3 Data Set')

ax = fig.add_subplot(324, projection='3d')
ax.plot3D(halfmoons[halfmoons_labels==1,0],halfmoons[halfmoons_labels==1,1], halfmoons[halfmoons_labels==1,2], '.r')
ax.plot3D(halfmoons[halfmoons_labels==2,0],halfmoons[halfmoons_labels==2,1], halfmoons[halfmoons_labels==2,2], '.g')
ax.plot3D(halfmoons[halfmoons_labels==3,0],halfmoons[halfmoons_labels==3,1], halfmoons[halfmoons_labels==3,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Halfmoons Data Set')

ax = fig.add_subplot(325, projection='3d')
ax.plot3D(swissroll[:,0],swissroll[:,1],swissroll[:,2], '.b')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('Swissroll Data Set')

ax = fig.add_subplot(326, projection='3d')
ax.plot3D(seeds[seeds_labels==1,3],seeds[seeds_labels==1,5], seeds[seeds_labels==1,2], '.r')
ax.plot3D(seeds[seeds_labels==2,3],seeds[seeds_labels==2,5], seeds[seeds_labels==2,2], '.g')
ax.plot3D(seeds[seeds_labels==3,3],seeds[seeds_labels==3,5], seeds[seeds_labels==3,2], '.b')
ax.set_xlabel('Dimension 4')
ax.set_ylabel('Dimension 6')
ax.set_zlabel('Dimension 3')
ax.set_title('Seeds Data Set')

plt.show()

# Perform PCA


if __name__ == '__main__':
	assignment05()