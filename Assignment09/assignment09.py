import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from math import sqrt

from ReadData import ReadData
from ExtractSamples import ExtractSamples
from ColorModels import ColorModels
from CrossValidation import CrossValidation
from RunKMeans import RunKMeans
from RunEM_GMM import RunEM_GMM
from RunCommands import RunFCM, RunPCM
from sklearn.metrics import confusion_matrix, roc_curve
from scipy.stats import multivariate_normal


data_train, labels, locations = ReadData() # Load all data
Data, ObjLabels = ExtractSamples(data_train, labels, locations) # extract objects
plt.close("all") # close all image plots
Data = Data/255 #Normalize pixel values to be between 0 and 1
Data_HSV, Data_YIQ, Data_HLS = ColorModels(Data, ObjLabels) # Transform RGB to different color spaces

DTrain, DVal, labelsTrain, labelsVal = CrossValidation(Data,ObjLabels,0.8,'RGB') #80% of data for training and 20% for testing
DTrain_HSV, DVal_HSV, labelsTrain_HSV, labelsVal_HSV = CrossValidation(Data_HSV,ObjLabels,0.8,'HSV') #80% of data for training and 20% for testing
DTrain_YIQ, DVal_YIQ, labelsTrain_YIQ, labelsVal_YIQ = CrossValidation(Data_YIQ,ObjLabels,0.8,'YIQ') #80% of data for training and 20% for testing
DTrain_HLS, DVal_HLS, labelsTrain_HLS, labelsVal_HLS = CrossValidation(Data_HLS,ObjLabels,0.8,'HLS') #80% of data for training and 20% for testing


#%% Perform K-Means

NumClusters = 4

centers, L          = RunKMeans(DTrain.T,labelsTrain,NumClusters,0,'RGB')
centers_HSV, L_HSV  = RunKMeans(DTrain_HSV.T,labelsTrain_HSV,NumClusters,0,'HSV')
centers_YIQ, L_YIQ  = RunKMeans(DTrain_YIQ.T,labelsTrain_YIQ,NumClusters,0,'YIQ')
centers_HLS, L_HLS  = RunKMeans(DTrain_HLS.T,labelsTrain_HLS,NumClusters,0,'HLS')
dist                = sp.spatial.distance.cdist(DVal.T, centers).T
dist_HSV            = sp.spatial.distance.cdist(DVal.T, centers_HSV).T
dist_YIQ            = sp.spatial.distance.cdist(DVal.T, centers_YIQ).T
dist_HLS            = sp.spatial.distance.cdist(DVal.T, centers_HLS).T
labelsTest          = np.argmin(dist, axis = 0)
labelsTest          += 1

labelsTest_HSV      = np.argmin(dist_HSV, axis = 0)
labelsTest_HSV      += 1

labelsTest_YIQ      = np.argmin(dist_YIQ, axis = 0)
labelsTest_YIQ      += 1

labelsTest_HLS      = np.argmin(dist_HLS, axis = 0)
labelsTest_HLS      += 1

print("---------------------------------------------")
print("---------------------------------------------")
print("K-Means")
print("---------------------------------------------")
print("Confusion Matrices in the Training Set")
print("---------------------------------------------")
print("RGB")
print(confusion_matrix(labelsTrain, L))
fig = plt.figure()
fpr, tpr, thresholds = roc_curve(labelsTrain, L/4, pos_label=4)
plt.plot(fpr,tpr,color='y',label='RGB')
print("---------------------------------------------")
print("HSV")
print(confusion_matrix(labelsTrain_HSV, L_HSV))
fpr, tpr, thresholds = roc_curve(labelsTrain, L_HSV/4, pos_label=4)
plt.plot(fpr,tpr,color='r',label='HSV')
print("---------------------------------------------")
print("YIQ")
print(confusion_matrix(labelsTrain_YIQ, L_YIQ))
fpr, tpr, thresholds = roc_curve(labelsTrain, L_YIQ/4, pos_label=4)
plt.plot(fpr,tpr,color='b',label='YIQ')
print("---------------------------------------------")
print("HLS")
print(confusion_matrix(labelsTrain_HLS, L_HLS))
fpr, tpr, thresholds = roc_curve(labelsTrain, L_HLS/4, pos_label=4)
plt.plot(fpr,tpr,color='k',label='HLS')
plt.title("ROC Curves using K-Means in Training")
plt.legend()
print("---------------------------------------------")



print("---------------------------------------------")
print("Confusion Matrices in the Testing Set")
print("---------------------------------------------")
print("RGB Validation")
fig = plt.figure()
print(confusion_matrix(labelsVal, labelsTest))
fpr, tpr, thresholds = roc_curve(labelsVal, labelsTest/4, pos_label=4)
plt.plot(fpr,tpr,color='y',label='RGB')
print("---------------------------------------------")


print("HSV Validation")
print(confusion_matrix(labelsVal_HSV, labelsTest_HSV))
fpr, tpr, thresholds = roc_curve(labelsVal_HSV,labelsTest_HSV/4, pos_label=4)
plt.plot(fpr,tpr,color='r',label='HSV')
print("---------------------------------------------")


print("YIQ Validation")
print(confusion_matrix(labelsVal_YIQ, labelsTest_YIQ))
fpr, tpr, thresholds = roc_curve(labelsVal_YIQ, labelsTest_YIQ/4, pos_label=4)
plt.plot(fpr,tpr,color='b',label='YIQ')
print("---------------------------------------------")


print("HLS Validation")
print(confusion_matrix(labelsVal_HLS, labelsTest_HLS))
fpr, tpr, thresholds = roc_curve(labelsVal_HLS, labelsTest_HLS/4, pos_label=4)
plt.plot(fpr,tpr,color='k',label='HLS')
plt.title("ROC Curves using K-Means in Validation")
plt.legend()
print("---------------------------------------------")

#%% Perform FCM
# UNCOMMENT FOR FCM SOLUTION: (data, labels, n_clusters, m, name)
centers, L_4          = RunFCM(DTrain.T,labelsTrain,NumClusters,2,'RGB')
centers_HSV, L_4_HSV  = RunFCM(DTrain_HSV.T,labelsTrain_HSV,NumClusters,2,'HSV')
centers_YIQ, L_4_YIQ  = RunFCM(DTrain_YIQ.T,labelsTrain_YIQ,NumClusters,2,'YIQ')
centers_HLS, L_4_HLS  = RunFCM(DTrain_HLS.T,labelsTrain_HLS,NumClusters,2,'HLS')

L = np.argmax(L_4, axis = 0)
L_HSV = np.argmax(L_4_HSV, axis = 0)
L_YIQ = np.argmax(L_4_YIQ, axis = 0)
L_HLS = np.argmax(L_4_HLS, axis = 0)

dist                = sp.spatial.distance.cdist(DVal.T, centers).T
dist_HSV            = sp.spatial.distance.cdist(DVal.T, centers_HSV).T
dist_YIQ            = sp.spatial.distance.cdist(DVal.T, centers_YIQ).T
dist_HLS            = sp.spatial.distance.cdist(DVal.T, centers_HLS).T

labelsTest          = np.argmin(dist, axis = 0)
labelsTest          += 1

labelsTest_HSV      = np.argmin(dist_HSV, axis = 0)
labelsTest_HSV      += 1

labelsTest_YIQ      = np.argmin(dist_YIQ, axis = 0)
labelsTest_YIQ      += 1

labelsTest_HLS      = np.argmin(dist_HLS, axis = 0)
labelsTest_HLS      += 1

print("---------------------------------------------")
print("---------------------------------------------")
print("FCM")
print("---------------------------------------------")
print("Confusion Matrices in the Training Set")
print("---------------------------------------------")
print("RGB")
print(confusion_matrix(labelsTrain, L))
fig = plt.figure()
fpr, tpr, thresholds = roc_curve(labelsTrain, L/4, pos_label=4)
plt.plot(fpr,tpr,color='y',label='RGB')


print("HSV")
print(confusion_matrix(labelsTrain_HSV, L_HSV))
fpr, tpr, thresholds = roc_curve(labelsTrain_HSV, L_HSV/4, pos_label=4)
plt.plot(fpr,tpr,color='r',label='HSV')
print("---------------------------------------------")


print("YIQ")
print(confusion_matrix(labelsTrain_YIQ, L_YIQ))
fpr, tpr, thresholds = roc_curve(labelsTrain_YIQ, L_YIQ/4, pos_label=4)
plt.plot(fpr,tpr,color='b',label='YIQ')
print("---------------------------------------------")


print("HLS")
print(confusion_matrix(labelsTrain_HLS, L_HLS))
fpr, tpr, thresholds = roc_curve(labelsTrain_HLS, L_HLS/4, pos_label=4)
plt.plot(fpr,tpr,color='k',label='HLS')
plt.title("ROC Curves using FCM in Training")
plt.legend()
print("---------------------------------------------")


print("---------------------------------------------")
print("Confusion Matrices in the Testing Set")
print("---------------------------------------------")
print("RGB Validation")
fig = plt.figure()
print(confusion_matrix(labelsVal, labelsTest))
fpr, tpr, thresholds = roc_curve(labelsVal, labelsTest/4, pos_label=4)
plt.plot(fpr,tpr,color='y',label='RGB')
print("---------------------------------------------")


print("HSV Validation")
print(confusion_matrix(labelsVal_HSV, labelsTest_HSV))
fpr, tpr, thresholds = roc_curve(labelsVal_HSV,labelsTest_HSV/4, pos_label=4)
plt.plot(fpr,tpr,color='r',label='HSV')
print("---------------------------------------------")


print("YIQ Validation")
print(confusion_matrix(labelsVal_YIQ, labelsTest_YIQ))
fpr, tpr, thresholds = roc_curve(labelsVal_YIQ, labelsTest_YIQ/4, pos_label=4)
plt.plot(fpr,tpr,color='b',label='YIQ')
print("---------------------------------------------")


print("HLS Validation")
print(confusion_matrix(labelsVal_HLS, labelsTest_HLS))
fpr, tpr, thresholds = roc_curve(labelsVal_HLS, labelsTest_HLS/4, pos_label=4)
plt.plot(fpr,tpr,color='k',label='HLS')
plt.title("ROC Curves using FCM in Validation")
plt.legend()
print("---------------------------------------------")

#%% Perform PCM
 
centers, L_4         = RunPCM(DTrain.T,labelsTrain.T,NumClusters,2, 0.5, 'RGB')
centers_HSV, L_4_HSV  = RunPCM(DTrain_HSV.T,labelsTrain_HSV,NumClusters,2, 0.5, 'HSV')
centers_YIQ, L_4_YIQ  = RunPCM(DTrain_YIQ.T,labelsTrain_YIQ,NumClusters,2, 0.5, 'YIQ')
centers_HLS, L_4_HLS  = RunPCM(DTrain_HLS.T,labelsTrain_HLS,NumClusters,2, 0.5, 'HLS')

L = np.argmax(L_4.T, axis = 0)
L_HSV = np.argmax(L_4_HSV.T, axis = 0)
L_YIQ = np.argmax(L_4_YIQ.T, axis = 0)
L_HLS = np.argmax(L_4_HLS.T, axis = 0)

dist                = sp.spatial.distance.cdist(DVal.T, centers)
dist_HSV            = sp.spatial.distance.cdist(DVal.T, centers_HSV)
dist_YIQ            = sp.spatial.distance.cdist(DVal.T, centers_YIQ)
dist_HLS            = sp.spatial.distance.cdist(DVal.T, centers_HLS)

labelsTest          = np.argmin(dist, axis = 0)
labelsTest          += 1

labelsTest_HSV      = np.argmin(dist_HSV, axis = 0)
labelsTest_HSV      += 1

labelsTest_YIQ      = np.argmin(dist_YIQ, axis = 0)
labelsTest_YIQ      += 1

labelsTest_HLS      = np.argmin(dist_HLS, axis = 0)
labelsTest_HLS      += 1

print("---------------------------------------------")
print("---------------------------------------------")
print("PCM")
print("---------------------------------------------")
print("Confusion Matrices in the Training Set")
print("---------------------------------------------")
print("RGB")
print(confusion_matrix(labelsTrain, L))
fig = plt.figure()
fpr, tpr, thresholds = roc_curve(labelsTrain, L/4, pos_label=4)
plt.plot(fpr,tpr,color='y',label='RGB')


print("HSV")
print(confusion_matrix(labelsTrain_HSV, L_HSV))
fpr, tpr, thresholds = roc_curve(labelsTrain_HSV, L_HSV/4, pos_label=4)
plt.plot(fpr,tpr,color='r',label='HSV')
print("---------------------------------------------")


print("YIQ")
print(confusion_matrix(labelsTrain_YIQ, L_YIQ))
fpr, tpr, thresholds = roc_curve(labelsTrain_YIQ, L_YIQ/4, pos_label=4)
plt.plot(fpr,tpr,color='b',label='YIQ')
print("---------------------------------------------")


print("HLS")
print(confusion_matrix(labelsTrain_HLS, L_HLS))
fpr, tpr, thresholds = roc_curve(labelsTrain_HLS, L_HLS/4, pos_label=4)
plt.plot(fpr,tpr,color='k',label='HLS')
plt.title("ROC Curves using K-Means in Training")
plt.legend()
print("---------------------------------------------")


print("---------------------------------------------")
print("Confusion Matrices in the Testing Set")
print("---------------------------------------------")
print("RGB Validation")
fig = plt.figure()
print(confusion_matrix(labelsVal, labelsTest))
fpr, tpr, thresholds = roc_curve(labelsVal, labelsTest/4, pos_label=4)
plt.plot(fpr,tpr,color='y',label='RGB')
print("---------------------------------------------")


print("HSV Validation")
print(confusion_matrix(labelsVal_HSV, labelsTest_HSV))
fpr, tpr, thresholds = roc_curve(labelsVal_HSV,labelsTest_HSV/4, pos_label=4)
plt.plot(fpr,tpr,color='r',label='HSV')
print("---------------------------------------------")


print("YIQ Validation")
print(confusion_matrix(labelsVal_YIQ, labelsTest_YIQ))
fpr, tpr, thresholds = roc_curve(labelsVal_YIQ, labelsTest_YIQ/4, pos_label=4)
plt.plot(fpr,tpr,color='b',label='YIQ')
print("---------------------------------------------")


print("HLS Validation")
print(confusion_matrix(labelsVal_HLS, labelsTest_HLS))
fpr, tpr, thresholds = roc_curve(labelsVal_HLS, labelsTest_HLS/4, pos_label=4)
plt.plot(fpr,tpr,color='k',label='HLS')
plt.title("ROC Curves using K-Means in Validation")
plt.legend()
print("---------------------------------------------")

#%% Perform EM with a Gaussian Mixture (Testing)

NumComponents = 4

Means, Sigs, Ps, pZ_X = RunEM_GMM(DTrain.T,labelsTrain,NumClusters,'RGB')
Means_HSV, Sigs_HSV, Ps_HSV, pZ_X_HSV = RunEM_GMM(DTrain_HSV.T,labelsTrain_HSV,NumClusters,'HSV')
Means_YIQ, Sigs_YIQ, Ps_YIQ, pZ_X_YIQ = RunEM_GMM(DTrain_YIQ.T,labelsTrain_YIQ,NumClusters,'YIQ')
Means_HLS, Sigs_HLS, Ps_HLS, pZ_X_HLS = RunEM_GMM(DTrain_HLS.T,labelsTrain_HLS,NumClusters,'HLS')

print("---------------------------------------------")
print("---------------------------------------------")
print("EM for GMM Model")
print("---------------------------------------------")
print("Confusion Matrices in the Training Set")
print("---------------------------------------------")
print("RGB")
fig = plt.figure()
l_pred = np.zeros((labelsTrain.shape))
for i in range(l_pred.shape[0]):
    l_pred[i] = np.array(np.where(pZ_X[i,:]==max(pZ_X[i,:])))[0][0] + 1
print(confusion_matrix(labelsTrain, l_pred))
fpr, tpr, thresholds = roc_curve(labelsTrain, l_pred/4, pos_label=4)
plt.plot(fpr,tpr,color='y',label='RGB')
print("---------------------------------------------")
print("HSV")
l_pred_HSV = np.zeros((labelsTrain.shape))
for i in range(l_pred.shape[0]):
    l_pred_HSV[i] = np.array(np.where(pZ_X_HSV[i,:]==max(pZ_X_HSV[i,:])))[0][0] + 1
print(confusion_matrix(labelsTrain_HSV, l_pred_HSV))
fpr, tpr, thresholds = roc_curve(labelsTrain, l_pred_HSV/4, pos_label=4)
plt.plot(fpr,tpr,color='r',label='HSV')
print("---------------------------------------------")
print("YIQ")
l_pred_YIQ = np.zeros((labelsTrain.shape))
for i in range(l_pred.shape[0]):
    l_pred_YIQ[i] = np.array(np.where(pZ_X_YIQ[i,:]==max(pZ_X_YIQ[i,:])))[0][0] + 1
print(confusion_matrix(labelsTrain_YIQ, l_pred_YIQ))
fpr, tpr, thresholds = roc_curve(labelsTrain, l_pred_YIQ/4, pos_label=4)
plt.plot(fpr,tpr,color='b',label='YIQ')
print("---------------------------------------------")
print("HLS")
l_pred_HLS = np.zeros((labelsTrain.shape))
for i in range(l_pred.shape[0]):
    l_pred_HLS[i] = np.array(np.where(pZ_X_HLS[i,:]==max(pZ_X_HLS[i,:])))[0][0] + 1
print(confusion_matrix(labelsTrain_HLS, l_pred_HLS))
fpr, tpr, thresholds = roc_curve(labelsTrain, l_pred_HLS/4, pos_label=4)
plt.plot(fpr,tpr,color='k',label='HLS')
plt.legend()
plt.title("ROC Curves using EM for GMM Models in Training")
print("---------------------------------------------")




#%% Perform EM with a Gaussian Mixture Validation Set (Fitting)

print("---------------------------------------------")
print("---------------------------------------------")
print("EM for GMM Model")
print("---------------------------------------------")
print("Confusion Matrices in the Validation Set")
print("---------------------------------------------")
print("RGB")
fig = plt.figure()
l_pred = np.zeros((labelsVal.shape))
for i in range(l_pred.shape[0]):
     g = np.zeros(4)
     for j in range(0,NumComponents):
         g[j] = multivariate_normal.pdf(DVal[:,i].T,Means[j,:],Sigs[:,:,j])*Ps[j]
         l_pred[i] = (g.argmax(axis=0) + 1)
         
print(confusion_matrix(labelsVal, l_pred))    
fpr, tpr, thresholds = roc_curve(labelsVal, l_pred/4, pos_label=4)
plt.plot(fpr,tpr,color='y',label='RGB')
print("---------------------------------------------")
print("HSV")
l_pred_HSV = np.zeros((labelsVal.shape))
for i in range(l_pred.shape[0]):
     g = np.zeros(4)
     for j in range(0,NumComponents):
         g[j] = multivariate_normal.pdf(DVal_HSV[:,i].T,Means_HSV[j,:],Sigs_HSV[:,:,j])*Ps_HSV[j]
         l_pred_HSV[i] = (g.argmax(axis=0) + 1)  
         
print(confusion_matrix(labelsVal_HSV, l_pred_HSV))
fpr, tpr, thresholds = roc_curve(labelsVal_HSV, l_pred_HSV/4, pos_label=4)
plt.plot(fpr,tpr,color='r',label='HSV')
print("---------------------------------------------")
print("YIQ")         
l_pred_YIQ = np.zeros((labelsVal.shape))
for i in range(l_pred.shape[0]):
     g = np.zeros(4)
     for j in range(0,NumComponents):
         g[j] = multivariate_normal.pdf(DVal_YIQ[:,i].T,Means_YIQ[j,:],Sigs_YIQ[:,:,j])*Ps_YIQ[j]
         l_pred_YIQ[i] = (g.argmax(axis=0) + 1)  
         
print(confusion_matrix(labelsVal_YIQ, l_pred_YIQ))
fpr, tpr, thresholds = roc_curve(labelsVal_YIQ, l_pred_YIQ/4, pos_label=4)
plt.plot(fpr,tpr,color='b',label='YIQ')
print("---------------------------------------------")
print("HLS")         
l_pred_HLS = np.zeros((labelsVal.shape))
for i in range(l_pred.shape[0]):
     g = np.zeros(4)
     for j in range(0,NumComponents):
         g[j] = multivariate_normal.pdf(DVal_HLS[:,i].T,Means_HLS[j,:],Sigs_HLS[:,:,j])*Ps_HLS[j]
         l_pred_HLS[i] = (g.argmax(axis=0) + 1)  
         
print(confusion_matrix(labelsVal_HLS, l_pred_HLS))
fpr, tpr, thresholds = roc_curve(labelsVal_HLS, l_pred_HLS/4, pos_label=4)
plt.plot(fpr,tpr,color='g',label='HLS')
plt.legend()
plt.title("ROC Curves using EM for GMM Models in Validation")
print("---------------------------------------------")

