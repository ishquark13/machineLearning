import numpy as np
import matplotlib.pyplot as plt

def ExtractSamples(image,labels,locations):

    N, M, D = np.shape(image)  # training data size NxMxD
    N_WC = sum(labels==1)           # number of white car objects
    N_RC = sum(labels==2)           # number of red car objects
    N_pools = sum(labels==3)        # number of pool objects
    N_ponds = sum(labels==4)        # number of pond objects
    
    # RED CAR PIXELS
    halo_radius = 2 # 2 pixel radius
    locations_RC = locations[labels==2,:]
    RC_samples = {}
    for i in range(N_RC):
        RC_samples["{0}".format(i)]=np.zeros(2,)
    for i in range(N_RC):
        centerX = locations_RC[i,0]
        centerY = locations_RC[i,1]
        if centerX - halo_radius != 0:
            jStart = centerX - halo_radius - 1
        else:
            jStart = 0
        if centerY - halo_radius != 0:
            kStart = centerY - halo_radius - 1
        else:
            kStart = 0
        for j in range(jStart,jStart+halo_radius*2+2):
            for k in range(kStart,kStart+halo_radius*2+2):
                if ((j-centerX)**2 + (k-centerY)**2 <= halo_radius**2):
                    RC_samples[str(i)] = np.vstack((RC_samples[str(i)],[j,k]))
        RC_samples[str(i)] = RC_samples[str(i)][1::,:]
    
    # WHITE CAR PIXELS
    halo_radius = 2 # 2 pixel radius
    locations_WC = locations[labels==1,:]
    WC_samples = {}
    for i in range(N_WC):
        WC_samples["{0}".format(i)]=np.zeros(2,)
    for i in range(N_WC):
        centerX = locations_WC[i,0]
        centerY = locations_WC[i,1]
        if centerX - halo_radius != 0:
            jStart = centerX - halo_radius - 1
        else:
            jStart = 0
        if centerY - halo_radius != 0:
            kStart = centerY - halo_radius - 1
        else:
            kStart = 0
        for j in range(jStart,jStart+halo_radius*2+2):
            for k in range(kStart,kStart+halo_radius*2+2):
                if ((j-centerX)**2 + (k-centerY)**2 < halo_radius**2):
                    WC_samples[str(i)] = np.vstack((WC_samples[str(i)],[j,k]))
        WC_samples[str(i)] = WC_samples[str(i)][1::,:]
    
    # POOL PIXELS
    halo_radius = [5,5,15,5,10,10,10,7] # radius of the halo
    locations_pools = locations[labels==3,:]
    Pool_samples = {}
    for i in range(N_pools):
        Pool_samples["{0}".format(i)]=np.zeros(2,)
    for i in range(N_pools):
        centerX = locations_pools[i,0]
        centerY = locations_pools[i,1]
        if centerX - halo_radius[i] != 0:
            jStart = centerX - halo_radius[i] - 1
        else:
            jStart = 0
        if centerY - halo_radius[i] != 0:
            kStart = centerY - halo_radius[i] - 1
        else:
            kStart = 0
        for j in range(jStart,jStart+halo_radius[i]*2+2):
            for k in range(kStart,kStart+halo_radius[i]*2+2):
                if ((j-centerX)**2 + (k-centerY)**2 < halo_radius[i]**2):
                    Pool_samples[str(i)]=np.vstack((Pool_samples[str(i)],[j,k]))
        Pool_samples[str(i)] = Pool_samples[str(i)][1::,:]
    
    #% Collect data samples
    ObjLabels = np.zeros(1)
    XXdata = np.zeros(1)
    YYdata = np.zeros(1)
    ZZdata = np.zeros(1)
    for i in range(N_pools):
        pixels = Pool_samples[str(i)].astype(int)
        xx = image[pixels[:,1],pixels[:,0],0]
        yy = image[pixels[:,1],pixels[:,0],1]
        zz = image[pixels[:,1],pixels[:,0],2]
        ObjLabels = np.hstack((ObjLabels,np.ones(xx.size)*3))
        XXdata = np.hstack((XXdata,xx))
        YYdata = np.hstack((YYdata,yy))
        ZZdata = np.hstack((ZZdata,zz))
    for i in range(N_RC):
        pixels = RC_samples[str(i)].astype(int)
        xx = image[pixels[:,1],pixels[:,0],0]
        yy = image[pixels[:,1],pixels[:,0],1]
        zz = image[pixels[:,1],pixels[:,0],2]
        ObjLabels = np.hstack((ObjLabels,np.ones(xx.size)*2))
        XXdata = np.hstack((XXdata,xx))
        YYdata = np.hstack((YYdata,yy))
        ZZdata = np.hstack((ZZdata,zz))
    for i in range(N_WC):
        pixels = WC_samples[str(i)].astype(int)
        xx = image[pixels[:,1],pixels[:,0],0]
        yy = image[pixels[:,1],pixels[:,0],1]
        zz = image[pixels[:,1],pixels[:,0],2]
        ObjLabels = np.hstack((ObjLabels,np.ones(xx.size)))
        XXdata = np.hstack((XXdata,xx))
        YYdata = np.hstack((YYdata,yy))
        ZZdata = np.hstack((ZZdata,zz))
    
    pond={}
    for i in range(1,N_ponds+1):
        pond["{0}".format(i)]=np.loadtxt("pond" + str(i) + ".txt")
    
    for i in range(1,N_ponds+1):
        locations = pond[str(i)].astype(int)
        locations = locations.astype(int)
        xx = image[locations[:,0],locations[:,1],0]
        yy = image[locations[:,0],locations[:,1],1]
        zz = image[locations[:,0],locations[:,1],2]
        XXdata = np.hstack((XXdata,xx))
        YYdata = np.hstack((YYdata,yy))
        ZZdata = np.hstack((ZZdata,zz))
        ObjLabels = np.hstack((ObjLabels,np.ones(xx.size)*4))
        
    ObjLabels = ObjLabels[1::]
    Data = np.array(([XXdata[1::],YYdata[1::],ZZdata[1::]]))
    
    '''
   fig = plt.figure()
    Pools_segmentation = np.zeros((N,M,D))
    for i in range(N_pools):
        pixels = Pool_samples[str(i)].astype(int)
        Pools_segmentation[pixels[:,1],pixels[:,0],:] = image[pixels[:,1],pixels[:,0],:]/255
    plt.imshow(Pools_segmentation)
    
    fig = plt.figure()
    RC_segmentation = np.zeros((N,M,D))
    for i in range(N_RC):
        pixels = RC_samples[str(i)].astype(int)
        RC_segmentation[pixels[:,1],pixels[:,0],:] = image[pixels[:,1],pixels[:,0],:]/255
    plt.imshow(RC_segmentation)
    
    fig = plt.figure()
    WC_segmentation = np.zeros((N,M,D))
    for i in range(N_WC):
        pixels = WC_samples[str(i)].astype(int)
        WC_segmentation[pixels[:,1],pixels[:,0],:] = image[pixels[:,1],pixels[:,0],:]/255
    plt.imshow(WC_segmentation)
    
    # Plot masks for all ponds
    for i in range(1,9):
        pixels = pond[str(i)].astype(int)
        pp = np.zeros((np.shape(image)))
        plt.figure()
        for j in range(3):
            pp[pixels[:,1],pixels[:,0],j] = image[pixels[:,1],pixels[:,0],j]/255
        plt.imshow(pp)
        plt.title('Pond '+str(i))
        # plt.show()        
        del pp
        '''
    return Data, ObjLabels

    
