from fcm import cmeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def RunFCM(data,labels,NumClusters,flag,name):

    cntr, u = cmeans(data, NumClusters)

    u = u + 1

	# fig = plt.figure()
	# ax = fig.add_subplot(121, projection='3d')
	# p1 = ax.scatter(data[:,0], data[:,1],data[:,2], c=labels) 
	# ax.set_title('Data in ' + name +' Color Model')
	# fig.colorbar(p1, ax=ax)
	# ax = fig.add_subplot(122, projection='3d')
	# p1 = ax.scatter(data[:,0],data[:,1],data[:,2],c=u)
	# ax.set_title('FCM in ' + name + ' Color Model')
	# fig.colorbar(p1, ax=ax)
	# plt.suptitle('FCM of Data in ' + name + ' Color Model')
	
    return cntr, u