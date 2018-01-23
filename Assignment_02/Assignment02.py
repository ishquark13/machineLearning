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
	#get w parameters
	#feature extract

	#allocate memory for array
	X = np.zeros(shape=(M,M))

	X = np.array([x**m for m in range(M+1)]).T
	#make sure size match 


	w = np.linalg.inv(X.T@X)@X.T@t

	#adding residue
	#lambda = 0.035
	#eye    = lambda*(np.identity(M+1))
	#w = np.linalg.inv(X.T@X+eye)@X.T@t
	return w


def fitdata2(x,t,M,l):
	'''fitdata(x,t,M): Fit a polynomial of order M to the data (x,t)'''	
	#This needs to be filled in
	#get w parameters
	#feature extract

	#allocate memory for array
	X = np.zeros(shape=(M,M))

	X = np.array([x**m for m in range(M+1)]).T
	#make sure size match 


	#w = np.linalg.inv(X.T@X)@X.T@t

	#adding residue
	lambduh = l
	eye     = lambduh*(np.identity(M+1))
	w       = np.linalg.inv(X.T@X+eye)@X.T@t
	return w	


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
	data_good  = generateRandData(10,-.5,.5,0)
	data_noise = generateUniformData(10,-.5,.5,0)

	data_total = np.transpose(np.add(data_good, data_noise))

	#print(data_total)

	#size should be 10,2

	#print(np.shape(data_total))



	x = np.array(data_total[:,0])
	t = np.array(data_total[:,1])

	xx = np.array(generateUniformData(10,-1,1,.209)).T
	t1 = np.array(xx[:,1])
	x1 = np.array(xx[:,0])

	#allocate memory for array
	#X = np.zeros(shape=(M,M))

	#X = np.array([x**m for m in range(M+1)]).T
	#make sure size match 

	'''
	print(np.shape(x))
	print(np.shape(X))
	print(X)
	'''

	#print(np.shape(t))
	#w = np.linalg.inv(X.T@X)@X.T@t
	 
	#print(w)
	#print(np.shape(w))


	 

	#generate new X_test data to see 
	# try with x and t having noise added or
	# try with x1 and t1 with randomly generated variance



#part1
# test for figs 1.4
# call 4 times for different M
w     = fitdata(x1,t1,0) # try with order as last params M
plotPoly(x1,t1,w,-1,1,[221])
w     = fitdata(x1,t1,1) # try with order as last params M
plotPoly(x1,t1,w,-1,1,[222])
w     = fitdata(x1,t1,3) # try with order as last params M
plotPoly(x1,t1,w,-1,1,[223])
w     = fitdata(x1,t1,9) # try with order as last params M
plotPoly(x1,t1,w,-1,1,[224])
plt.show()

#part 2
# number of data points
xx = np.array(generateUniformData(15,-1,1,.209)).T
t1 = np.array(xx[:,1])
x1 = np.array(xx[:,0])
w     = fitdata(x1,t1,0) # try with order as last params M
plotPoly(x1,t1,w,-1,1,[211])

xx = np.array(generateUniformData(100,-1,1,.209)).T
t1 = np.array(xx[:,1])
x1 = np.array(xx[:,0])
w     = fitdata(x1,t1,0) # try with order as last params M
plotPoly(x1,t1,w,-1,1,[212])
plt.show()

# part 3
# increase regularization
xx = np.array(generateUniformData(10,-1,1,.209)).T
t1 = np.array(xx[:,1])
x1 = np.array(xx[:,0])
w_reg = fitdata2(x1,t1,10, 0.0022) #small lambda
plotPoly(x1,t1,w_reg,-1,1,[211])
w_reg = fitdata2(x1,t1,10, 1) #large
plotPoly(x1,t1,w_reg,-1,1,[212])
plt.show()

# part 4 
# show Erms


M = 9

w = fitdata(x1,t1,9) # try with order as last params M
#generate RMS vals
X1 = np.array([x1**m for m in range(M+1)]).T
y_train = X1@w.T

trainRms = 0
print(y_train)
print(t1)

for m in range(t1.size):
	trainRms = trainRms + ((y_train-t1)**2)

trainRms = np.array(np.sqrt((trainRms/M+1)))

#ensure overlap

xx2 = np.array(generateUniformData(10,-.5,1.5,.209)).T
x2  = np.array(xx2[:,0])

X2  = np.array([x2**m for m in range(M+1)]).T
y_test = X2@w.T

testRms = 0

for m in range(t1.size):
	testRms = testRms + ((y_test-t1)**2)

testRms = np.array(np.sqrt((testRms/M+1)))

plt.plot(trainRms)
plt.plot(testRms)	
plt.show()


