''' Adding this comment as
a proof that my terminal is able 
to push to github properly
'''

import math 
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

def assignment01():

	''' Assignment 01 for Introduction to Machine Learning, Sp 2018
		Introduction to Python and Git assignment
	''' 


'''
Git tutorial
git init
git status
git add johns.txt
git status
git commit -m "johning"
git add '*.txt' # adds all the files to repo with the file extension
git commit - "add all txt files"
git log
git remote add origin https://github.com/try-git/try_git.git
git push -u origin master
git pull origin master
git diff HEAD
git add folder/name.txt
git diff --staged
git reset folder/name.txt
git checkout -- johns.txt
git branch clean_up
git checkout clean_up
git rm '*.txt'
git commit -m 'clean all the names'
git checkout master
git merge clean_up
git branch -d clean_up
git push
'''
# load data
test = np.array(np.load("dataX.npy"))
print(np.shape((test)))
# git clone, add, commit, push, pull
# compute mean 2D vector

test_mean = np.array(np.mean(test[:,0]), np.mean(test[:,1]))

#sum(l)/float(len(l))

# subtract mean from each data point
test_sub = np.array(test-test_mean)

# take L^2 norm of each data point
# np.linalg.norm

'''test_norm = np.array(math.sqrt((test[0,:]**2)+(test[1,:]**2))) 
for i in range(len(test)):
	for j in range(len(test))
		math.sqrt(i**2+j**2)
print(test_norm)'''

test_square      = np.array(test_sub**2)

test_square_sum  = np.array(test_square[:,0] + test_square[:,1])
test_sqrt        = np.array(np.sqrt(test_square_sum))


test_temp1       = test_sub[:,0] / test_sqrt
test_temp2       = test_sub[:,1] / test_sqrt




test_comp        = np.transpose(np.vstack((test_temp1,test_temp2)))

print(np.shape(test_comp))



fig               = plt.figure()

fig.subplots_adjust(hspace=.5)
plt.subplot(3,1,1)
plt.title('Initial Data')
plt.scatter(test[:,0], test[:,1])
plt.subplot(3,1,2)
plt.title('Mean-subtracted Data')
plt.scatter(test_sub[:,0], test_sub[:,1])
plt.subplot(3,1,3)
plt.title('Normalized Data')
plt.scatter(test_comp[:,0], test_comp[:,1])
plt.show()


###### PART TWO ########


if __name__ == '__main__':
	assignment01()
