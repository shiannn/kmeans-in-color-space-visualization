from sklearn.metrics import pairwise_distances
import numpy as np
import sys
import cv2

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
    if dyn==1: #greedy insertion, possible at all points
        k=1
        M=np.mean(X,axis=0)  #mean of 3 colors(RGB)
        K=pairwise_distances(X.T,metric = "sqeuclidean")
        L=X        #source img
        #print(M)
    elif dyn==2: 
        # use kd-tree results as means
        k=kmax
        M=kdtree(X,np.arange(1,n).T,[],1.5*n/k)
        #[1:n] row vector, become column vector
        nb=M.shape[1] 
        dyn=0
    elif dyn==3:
        L=kdtree(X,np.arange(1,n).T,[],1.5*n/bs)
        nb=L.shape[1]
        k=1
        M=mean(X,axis=0)
        K=pairwise_distances(X.T,L.T,metric = "sqeuclidean")
    elif dyn==4:
        k=1
        M=mean(X,axis=0)
        K=pairwise_distances(X.T,metric = "sqeuclidean")
        L=X
    else:# use random subset of data as means
        k=kmax
        tmp=np.random.permutation(n)
        M=X[tmp[0:k-1],:] #tmp[1:k]
    
    realmax=sys.float_info.max
    while(k<=kmax):
        kill=np.array([])
        # M is mean (16x3)
        # X is all the points with RGB 3 value (154401 x 3)
        Dist = pairwise_distances(M,X)  #squared Euclidean distances to means; Dist (16 x 154401)
        # Dist[i,j] 表示 第i個平均 mu 和第j個 X-pixel 的距離
        #Dist 是 16x154401 (k x pixel數量)
        Dist = Dist.T
        # Dist[i,j] 表示 第i個X-pixel 和第j個 平均 mu 的距離
        #Dist 是 154401x16 (pixel數量 x k)
        Dwin = np.amin(Dist,axis=1)
        # Dwin 找出每個點和16個cluster中心最短的距離   (154401 x 1)
        Iwin = np.argmin(Dist,axis=1)
        # Iwin 找出每個點和16個cluster中心的哪一個最短 (154401 x 1 且 值只可能是 1~16)
        P = Iwin

        # error measures and mean updates
        Wnew = np.sum(Dwin,axis=0) #(所有點的 Dwin 全部加起來)

        #update VQ's
        #更新cluster的中心
        testSum = 0
        for i in range(len(M)):
            #16個cluster中心
            #遍歷所有index i 找出所有碰到中心i者
            I = np.argwhere(Iwin==i)
            testSum += len(I)
            #歸在i群的所有人
            #I是一個 2D array，會使記下所有座標 (I 的值是 1~154401)
            if I.shape[0] > d:
                #更新 cluster 中心 i
                #X 是 nxd 的 矩陣
                #找出所有 X 中落在 i 群的點，將其pixel value求平均
                temp = [X[pos[0]] for pos in I]
                M[i,:] = np.mean(temp,axis=0)
            elif killing==1:
                kill=np.append(kill,i)
        print(testSum)
        k=k+1







    [Er,M, nb, P] = [0,0,0,0]
    return [Er,M, nb, P]

if(__name__=='__main__'):
    img = np.load('Flower.npy')
    [m,n,d] = img.shape
    X = np.reshape(img,(m*n,d))
    [T,kmax,dyn,bs, killing, pl]=[0,16,0,0,0,0]
    kmeansO(X,T,kmax,dyn,bs, killing, pl)