import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

def ReadData():
    # Load Data
    mat = spio.loadmat('data_train.mat')
    data_train = mat.get('data_train')
    
    # Load Object Locations and Labels
    indices = np.loadtxt('labels.txt')
    locations = np.array((indices[:,0:2].astype(int)))
    labels = indices[:,2]
    # 1 - White Cars, 2 - Red Cars, 3 - Pools, 4 - Ponds
    no_labels = np.unique(labels)
    
    # Plot Training Image with Objects center-point marked
    '''
    fig = plt.figure()
    colors = ['w','r','b','g']
    ll = []
    plt.imshow(data_train)
    for i in range(len(no_labels)):
        lbl = plt.scatter(locations[labels==(i+1),0],locations[labels==(i+1),1],c=colors[i])
        ll = np.append(ll,lbl)
    plt.legend(ll,['White Cars','Red Cars','Pools','Ponds'])
    plt.title('Training Data')
    '''
    # plt.show()

    return data_train, labels, locations