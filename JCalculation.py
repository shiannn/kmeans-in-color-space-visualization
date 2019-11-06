import numpy as np
from sklearn.metrics import pairwise_distances

def JCalculation(class_map, M, St):
    [m, n] = class_map.shape
    N = m*n
    Sw = 0
    for l in range(0, np.max(class_map)+1):
        #print(l)
        #print(class_map)
        [m,n] = np.where(class_map == l)
        # mean vector of the vectors with class label l
        if len(m)==0: 
            continue
        m = m+1
        n = n+1
        m_l = [np.mean(m), np.mean(n)]
        m1 = np.tile(m_l, (len(m),1))
        
        mn = np.array([np.transpose(m), np.transpose(n)])
        mn = np.transpose(mn)
        mn = mn[mn[:,1].argsort(kind='stable')]
        
        #print(m1)
        #print(mn)
        K=pairwise_distances(mn,m1,metric = "sqeuclidean")
        Dist1 = np.sum(np.diag(K))
        Sw = Sw + Dist1
    
    J = (St-Sw)/Sw
    return J