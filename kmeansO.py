from sklearn.metrics import pairwise_distances
import numpy as np

def kmeansO(X,T,kmax,dyn,bs, killing, pl):
    Er = []
    TEr = []  
    #error monitoring
    #X is input img    #T is []
    [n,d] = X.shape
    #n pixels 以及 d 顏色
        #print(n,d)
    P = np.zeros((n,1))
    #n pixels
    Threshold = 1e-4 #0.0001
    #用來判斷是否 convergence 的 Threshold
        #print(Threshold)
    nb = 0
    #initialize
    if dyn==1:
        k=1
        M=np.mean(X,axis=0)  #mean of 3 colors(RGB)
        K=pairwise_distances(X.T,metric = "sqeuclidean")
        L=X        #source img
        print(M)



    [Er,M, nb, P] = [0,0,0,0]
    return [Er,M, nb, P]

if(__name__=='__main__'):
    X = [[1,2,3],[4,5,6],[7,8,9],[2,3,4],[5,2,7]]
    X = np.array(X)
    [T,kmax,dyn,bs, killing, pl]=[0,0,1,0,0,0]
    kmeansO(X,T,kmax,dyn,bs, killing, pl)