#Blank starter file for assignment04.py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
#generate data with known mean and variance

	mu,sigma = 0.69, 2
data      = np.random.normal(mu, sigma, 1000)
print('Max number of data points ' + str(len(data)))

#sample per sample basis N = 1
#prior mean and variance denoted by _o
mu_o      = 0.3
sigma_o   = 1


#allocate data
result0 = []
result1 = []
result2 = []
#to plot in the future the sample versus N
#have input for the user to change mu_o and sigma_o

for i in range(0,len(data)):
	mu_maxL = np.mean(data[i])
	#formula for prior
	mu_n    = (sigma**2/(sigma_o**2 + sigma**2))*mu_o + (sigma_o**2/(sigma_o**2 + sigma**2))*mu_maxL
	sigma_n = np.sqrt(sigma_o**2 * sigma**2/ (sigma**2 + sigma_o**2))

	#print(mu_n)
	#append data
	result0.append(data[i])
	result1.append(mu_o)
	result2.append(sigma_o)

	print('MLE:' + str(sum(result0)/len(result0)))	
	print('MAP:' + str(mu_n))
	mu_o      = mu_n
	sigma_o   = sigma_n
	input("Press enter to continue...\n")
	# mu_o      = mu_n
	# sigma_o   = sigma_n

'''\=
#subplot on different axes
plt.figure(1)
plt.plot(result1)
plt.xlabel('Number of Iterations', fontsize=12)
plt.ylabel('Inferred Bayesian Mean', fontsize=12)
plt.show()

plt.figure(2)
plt.plot(result2,'g')
plt.xlabel('Number of Iterations', fontsize=12)
plt.ylabel('Posterior Variance', fontsize=12)
plt.show()
'''



