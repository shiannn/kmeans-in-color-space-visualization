import numpy as np
from kmeansO import kmeansO

def RegionGrowing(img):
    [m,n,d] = img.shape
    X = np.reshape(img,(m*n,d))
    X = X.astype(np.float64)
    print(X[:,0],X[:,1],X[:,2])
    [tmp,M,tmp2,P] = kmeansO(X,[],16,0,0,0,0)
    return [0,0,0,0]