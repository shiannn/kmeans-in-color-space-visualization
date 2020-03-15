import numpy as np

def ValleyD(JI, scale, uj, sj):
    [m, n] = JI.shape
    a = [-0.6, -0.4, -0.2, 0, 0.2, 0,4]
    #valley size has to be larger than minsize under scale 1-4.
    minsize = [32, 128, 512, 2048]
    #try a value one by one to find the value gives the most number of valleys
    scale = minsize[scale]
    MaxVSize = 0
    ValleyI = zeros(m,n)
    for i = range(1,len(a)+1):
        TJ = uj + a[i]*sj
        #candidate valley point (< TJ) == 1
        VP = np.zeros(m,n)
        VP[JI <= TJ] = 1
        # 4-connectivity => candidate valley (large size segments with low J
        # value
    return ValleyI