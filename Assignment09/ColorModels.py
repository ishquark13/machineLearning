import numpy as np
import colorsys

def ColorModels(Data, ObjLabels):
    
    D,N = Data.shape
    Data_HSV = np.zeros((D,N)) # convert RGB points to HSV color model
    Data_YIQ = np.zeros((D,N)) # convert RGB points to YIQ color model
    Data_HLS = np.zeros((D,N)) # convert RGB points to HSL color model
    for i in range(N):
        Data_HSV[:,i] = colorsys.rgb_to_hsv(Data[0,i],Data[1,i],Data[2,i])
        Data_YIQ[:,i] = colorsys.rgb_to_yiq(Data[0,i],Data[1,i],Data[2,i])
        Data_HLS[:,i] = colorsys.rgb_to_hls(Data[0,i],Data[1,i],Data[2,i])
    
    return Data_HSV, Data_YIQ, Data_HLS
