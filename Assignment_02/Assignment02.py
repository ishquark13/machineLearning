import numpy as np
import matplotlib.pyplot as plt
import math 
import textwrap


def generateUniformData(N, l, u, gVar):
	'''generateUniformData(N, l, u, gVar): Generate N uniformly spaced data points in the range [l,u) with zero-mean Gaussian random noise with variance gVar'''
	# x = np.random.uniform(l,u,N)
	step = (u-l)/(N);
	x = np.arange(l+step/2,u+step/2,step)
	e = np.random.normal(0,gVar,N)
	t = np.sin(2*math.pi*x) + e
	return x,t

def generateRandData(N, l, u, gVar):
	'''generateRandData(N, l, u, gVar): Generate N uniformly random data points in the range [l,u) with zero-mean Gaussian random noise with variance gVar'''
	x = np.random.uniform(l,u,N)
	e = np.random.normal(0,gVar,N)
	t = np.sin(2*math.pi*x) + e
	return x,t

def fitdata(x,t,M):
	'''fitdata(x,t,M): Fit a polynomial of order M to the data (x,t)'''	
	#This needs to be filled in

def plotPoly(x,t,w,l,u,subplotloc):
	'''plotPoly(x,t,w,l,u,subplotloc): Plot data (x,t) and the polynomial with parameters w across the range [l,u) in a sub figure at location subplotloc'''
	xrange = np.arange(l,u,0.001)  #get equally spaced points in the xrange
	y = np.sin(2*math.pi*xrange) #compute the true function value
	X = np.array([xrange**m for m in range(w.size)]).T
	esty = X@w #compute the predicted value

	#plot everything
	plt.subplot(*subplotloc) #identify the subplot to use
	plt.tight_layout()
	p1 = plt.plot(xrange, y, 'g') #plot true value
	p2 = plt.plot(x, t, 'bo') #plot training data
	p3 = plt.plot(xrange, esty, 'r') #plot estimated value

	#add title, legend and axes labels
	plt.ylabel('t') #label x and y axes
	plt.xlabel('x')
	plt.rcParams["axes.titlesize"] = 10
	myTitle = 'Plot of data, true function, and estimated polynomial with order M = ' + str(w.size-1) + ' and N =' + str(x.size)
	fig.add_subplot(*subplotloc).set_title("\n".join(textwrap.wrap(myTitle, 50)))
	plt.legend((p1[0],p2[0],p3[0]),('True Function', 'Training Data', 'Estimated\nPolynomial'), fontsize=6)


if __name__ == '__main__':
	fig = plt.figure()

	#This needs to be filled in

