import matplotlib.pyplot as plt
import numpy as np

from ReadData import ReadData
from ExtractSamples import ExtractSamples
from ColorModels import ColorModels
from CrossValidation import CrossValidation
from RunKMeans import RunKMeans
from RunEM_GMM import RunEM_GMM
from RunFCM import RunFCM
from RunPCM import RunPCM
from sklearn.metrics import confusion_matrix, roc_curve

data_train, labels, locations = ReadData() # Load all data
Data, ObjLabels = ExtractSamples(data_train, labels, locations) # extract objects
plt.close("all") # close all image plots
Data = Data/255 #Normalize pixel values to be between 0 and 1
Data_HSV, Data_YIQ, Data_HLS = ColorModels(Data, ObjLabels) # Transform RGB to different color spaces

DTrain, DVal, labelsTrain, labelsVal = CrossValidation(Data,ObjLabels,0.8,'RGB') #80% of data for training and 20% for testing
DTrain_HSV, DVal_HSV, labelsTrain_HSV, labelsVal_HSV = CrossValidation(Data_HSV,ObjLabels,0.8,'HSV') #80% of data for training and 20% for testing
DTrain_YIQ, DVal_YIQ, labelsTrain_YIQ, labelsVal_YIQ = CrossValidation(Data_YIQ,ObjLabels,0.8,'YIQ') #80% of data for training and 20% for testing
DTrain_HLS, DVal_HLS, labelsTrain_HLS, labelsVal_HLS = CrossValidation(Data_HLS,ObjLabels,0.8,'HLS') #80% of data for training and 20% for testing
print(np.shape(DTrain))

#%% Perform K-Means

NumClusters = 4

centers, L = RunKMeans(DTrain.T,labelsTrain,NumClusters,0,'RGB')
centers_HSV, L_HSV = RunKMeans(DTrain_HSV.T,labelsTrain_HSV,NumClusters,0,'HSV')
centers_YIQ, L_YIQ = RunKMeans(DTrain_YIQ.T,labelsTrain_YIQ,NumClusters,0,'YIQ')
centers_HLS, L_HLS = RunKMeans(DTrain_HLS.T,labelsTrain_HLS,NumClusters,0,'HLS')

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


#%% Perform EM with a Gaussian Mixture

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


#%% Perform fcm 

NumClusters = 4

cntr, u = RunFCM(DTrain,labelsTrain,NumClusters,0,'RGB')
cntr_HSV, u_HSV = RunFCM(DTrain_HSV,labelsTrain_HSV,NumClusters,0,'HSV')
cntr_YIQ, u_YIQ = RunFCM(DTrain_YIQ,labelsTrain_YIQ,NumClusters,0,'YIQ')
cntr_HLS, u_HLS = RunFCM(DTrain_HLS,labelsTrain_HLS,NumClusters,0,'HLS')

print("---------------------------------------------")
print("---------------------------------------------")
print("FCM")
print("---------------------------------------------")
print("Confusion Matrices in the Training Set")
print("---------------------------------------------")
print("RGB")
print(confusion_matrix(labelsTrain, u))
fig = plt.figure()
fpr, tpr, thresholds = roc_curve(labelsTrain, u/4, pos_label=4)
plt.plot(fpr,tpr,color='y',label='RGB')
print("---------------------------------------------")
print("HSV")
print(confusion_matrix(labelsTrain_HSV, u_HSV))
fpr, tpr, thresholds = roc_curve(labelsTrain, u_HSV/4, pos_label=4)
plt.plot(fpr,tpr,color='r',label='HSV')
print("---------------------------------------------")
print("YIQ")
print(confusion_matrix(labelsTrain_YIQ, u_YIQ))
fpr, tpr, thresholds = roc_curve(labelsTrain, u_YIQ/4, pos_label=4)
plt.plot(fpr,tpr,color='b',label='YIQ')
print("---------------------------------------------")
print("HLS")
print(confusion_matrix(labelsTrain_HLS, u_HLS))
fpr, tpr, thresholds = roc_curve(labelsTrain, u_HLS/4, pos_label=4)
plt.plot(fpr,tpr,color='k',label='HLS')
plt.title("ROC Curves using K-Means in Training")
plt.legend()
print("---------------------------------------------")



#%% Perform fcm 

NumClusters = 4

centers, U = RunPCM(DTrain,labelsTrain,NumClusters,0,'RGB')
centers_HSV, U_HSV = RunPCM(DTrain_HSV,labelsTrain_HSV,NumClusters,0,'HSV')
centers_YIQ, U_YIQ = RunPCM(DTrain_YIQ,labelsTrain_YIQ,NumClusters,0,'YIQ')
centers_HLS, U_HLS = RunPCM(DTrain_HLS,labelsTrain_HLS,NumClusters,0,'HLS')

print("---------------------------------------------")
print("---------------------------------------------")
print("PCM")
print("---------------------------------------------")
print("Confusion Matrices in the Training Set")
print("---------------------------------------------")
print("RGB")
print(confusion_matrix(labelsTrain, U))
fig = plt.figure()
fpr, tpr, thresholds = roc_curve(labelsTrain, U/4, pos_label=4)
plt.plot(fpr,tpr,color='y',label='RGB')
print("---------------------------------------------")
print("HSV")
print(confusion_matrix(labelsTrain_HSV, U_HSV))
fpr, tpr, thresholds = roc_curve(labelsTrain, U_HSV/4, pos_label=4)
plt.plot(fpr,tpr,color='r',label='HSV')
print("---------------------------------------------")
print("YIQ")
print(confusion_matrix(labelsTrain_YIQ, U_YIQ))
fpr, tpr, thresholds = roc_curve(labelsTrain, U_YIQ/4, pos_label=4)
plt.plot(fpr,tpr,color='b',label='YIQ')
print("---------------------------------------------")
print("HLS")
print(confusion_matrix(labelsTrain_HLS, U_HLS))
fpr, tpr, thresholds = roc_curve(labelsTrain, U_HLS/4, pos_label=4)
plt.plot(fpr,tpr,color='k',label='HLS')
plt.title("ROC Curves using K-Means in Training")
plt.legend()
print("---------------------------------------------")