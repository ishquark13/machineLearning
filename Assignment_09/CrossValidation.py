import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def CrossValidation(data,labels,Ntrain,name):
    D,N = data.shape
    # 1 - White Cars, 2 - Red Cars, 3 - Pools, 4 - Ponds
    N_WC = sum(labels==1)
    N_RC = sum(labels==2)
    N_pools = sum(labels==3)
    N_ponds = sum(labels==4)
    
    TrainingIndices_WC = random.sample(list(range(0, N_WC)), round(Ntrain*N_WC).astype(int))
    ValidationIndices_WC = list(set(list(range(0, N_WC)))-set(TrainingIndices_WC))
    TrainingIndices_RC = random.sample(list(range(0, N_RC)), round(Ntrain*N_RC).astype(int))
    ValidationIndices_RC = list(set(list(range(0, N_RC)))-set(TrainingIndices_RC))
    TrainingIndices_pools = random.sample(list(range(0, N_pools)), round(Ntrain*N_pools).astype(int))
    ValidationIndices_pools = list(set(list(range(0, N_pools)))-set(TrainingIndices_pools))
    TrainingIndices_ponds = random.sample(list(range(0, N_ponds)), round(Ntrain*N_ponds).astype(int))
    ValidationIndices_ponds = list(set(list(range(0, N_ponds)))-set(TrainingIndices_ponds))
    
    DataTraining = data[:,TrainingIndices_WC]
    DataValidation = data[:,ValidationIndices_WC]
    labelsTraining = labels[TrainingIndices_WC]
    labelsValidation = labels[ValidationIndices_WC]
    
    DataTraining = np.hstack((DataTraining,data[:,TrainingIndices_RC]))
    DataValidation = np.hstack((DataValidation,data[:,ValidationIndices_RC]))
    labelsTraining = np.hstack((labelsTraining,labels[TrainingIndices_RC]))
    labelsValidation = np.hstack((labelsValidation,labels[ValidationIndices_RC]))
    
    DataTraining = np.hstack((DataTraining,data[:,TrainingIndices_pools]))
    DataValidation = np.hstack((DataValidation,data[:,ValidationIndices_pools]))
    labelsTraining = np.hstack((labelsTraining,labels[TrainingIndices_pools]))
    labelsValidation = np.hstack((labelsValidation,labels[ValidationIndices_pools]))
    
    DataTraining = np.hstack((DataTraining,data[:,TrainingIndices_ponds]))
    DataValidation = np.hstack((DataValidation,data[:,ValidationIndices_ponds]))
    labelsTraining = np.hstack((labelsTraining,labels[TrainingIndices_ponds]))
    labelsValidation = np.hstack((labelsValidation,labels[ValidationIndices_ponds]))
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(DataTraining[0,labelsTraining==1],DataTraining[1,labelsTraining==1],DataTraining[2,labelsTraining==1], c='y', label='White Cars')
    ax.scatter(DataTraining[0,labelsTraining==2],DataTraining[1,labelsTraining==2],DataTraining[2,labelsTraining==2], c='r', label='Red Cars')
    ax.scatter(DataTraining[0,labelsTraining==3],DataTraining[1,labelsTraining==3],DataTraining[2,labelsTraining==3], c='b', label='Pools')
    ax.scatter(DataTraining[0,labelsTraining==4],DataTraining[1,labelsTraining==4],DataTraining[2,labelsTraining==4], c='k', label='Ponds')
    ax.set_xlabel('Dimension 1 (Red Channel)')
    ax.set_ylabel('Dimension 2 (Green Channel)')
    ax.set_zlabel('Dimension 3 (Blue Channel)')
    ax.set_title('Data Objects in '+ name + ' Training Set')
    plt.legend()
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(DataValidation[0,labelsValidation==1],DataValidation[1,labelsValidation==1],DataValidation[2,labelsValidation==1], c='y', label='White Cars')
    ax.scatter(DataValidation[0,labelsValidation==2],DataValidation[1,labelsValidation==2],DataValidation[2,labelsValidation==2], c='r', label='Red Cars')
    ax.scatter(DataValidation[0,labelsValidation==3],DataValidation[1,labelsValidation==3],DataValidation[2,labelsValidation==3], c='b', label='Pools')
    ax.scatter(DataValidation[0,labelsValidation==4],DataValidation[1,labelsValidation==4],DataValidation[2,labelsValidation==4], c='k', label='Ponds')
    ax.set_xlabel('Dimension 1 (Red Channel)')
    ax.set_ylabel('Dimension 2 (Green Channel)')
    ax.set_zlabel('Dimension 3 (Blue Channel)')
    ax.set_title('Data Objects in '+ name + ' the Validation Set')
    plt.legend()
    plt.show()
    
    return DataTraining, DataValidation, labelsTraining, labelsValidation
    
    
    
    
    