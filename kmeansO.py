from sklearn.metrics import pairwise_distances
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from pylab import *

def kmeansO(X,T,kmax,dyn,bs, killing, pl,img):
    Er = np.array([])
    TEr = np.array([])
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
        M=X[tmp[0:k],:] #tmp[1:k]
    
    realmax=sys.float_info.max
    Wold = realmax

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.show(block=False)

    while(k<=kmax):
        kill=np.array([])
        # M is mean (16x3)
        # X is all the points with RGB 3 value (154401 x 3)
        Dist = pairwise_distances(M,X,metric = "sqeuclidean")  #squared Euclidean distances to means; Dist (16 x 154401)
        #print(Dist.shape)
        #Dist = np.multiply(Dist,Dist)
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

        #print('Dwin shape',Dwin.shape)
        #print('Dwin ',Dwin)
        # error measures and mean updates
        Wnew = np.sum(Dwin) #(所有點的 Dwin 全部加起來)
        print('Wnew',Wnew)
        #update VQ's
        #更新cluster的中心
        testSum = 0
        for i in range(len(M)):
            #16個cluster中心
            #遍歷所有index i 找出所有碰到中心i者
            Cluster_i = np.argwhere(Iwin==i)
            Cluster_i = Cluster_i.flatten()
            testSum += len(Cluster_i)
            #歸在i群的所有人
            #I是一個 2D array，會使記下所有座標 (I 的值是 1~154401)
            if len(Cluster_i) > d:
                #更新 cluster 中心 i
                #X 是 nxd 的 矩陣
                #找出所有 X 中落在 i 群的點，將其pixel value求平均
                temp = [X[pos] for pos in Cluster_i]
                M[i,:] = np.mean(temp,axis=0)
            elif killing==1:
                kill=np.append(kill,i)
        #print('testSum',testSum)
        if 1-Wnew/Wold < Threshold*(10-9*(k==kmax)):
            # Wnew 和 Wold 相差太近就做
            print('dyn',dyn)
            if dyn & (k < kmax):
                if dyn == 4:
                    best_Er = Wnew
                    for i in range(n):
                        Wold = np.inf
       	                Wtmp = Wnew
                        #print(M.shape)
                        #Mtmp = [M; X(i,:)]
                        Mtmp = np.vstack((M, X[i,:]))
                        while (1-Wtmp/Wold) > Threshold*10:
                            Wold = Wtmp
                            Dist = pairwise_distances(Mtmp,X,metric = "sqeuclidean")
                            #Dist = sqdist(Mtmp',X')
                            #[Dwin,Iwin] = min(Dist',[],2)
                            Dist = Dist.T
                            Dwin = np.amin(Dist,axis=1)
                            Iwin = np.argmin(Dist,axis=1)
                            #Wtmp = sum(Dwin)
                            Wtmp = np.sum(Dwin,axis=0)
                            for i in range(len(Mtmp)):
                                I = np.argwhere(Iwin==i)
                                if len(I)>d:
                                    #Mtmp(i,:) = mean(X(I,:))
                                    temp = [X[pos[0]] for pos in I]
                                    Mtmp[i,:] = np.mean(temp,axis=0)
                        if Wtmp < best_Er:
                            best_M = Mtmp
                            best_Er = Wtmp
                    M = best_M
                    Wnew = best_Er
                    if len(T)!=0:
                        tmp = pairwise_distances(T,M,metric = "sqeuclidean")
                        #tmp會是 拿T去掃M(T和每個M的距離) 然後 T有M個向量 M有N個向量
                        #M有N個點 每個都是RGB
                        #tmp就會生出 MxN 的矩陣

                        #tmp=sqdist(T',M')
                        #sqdist 一定要是行向量才能操作
                        TEr = np.vstack((TEr, np.sum(np.amin(tmp,axis=1),axis=0)))
                        #TEr是純數
                        #TEr=[TEr; sum(min(tmp,[],2))]
                    Er=np.vstack((Er, Wnew))
                    k = k+1
                else:
                    # try to add a new cluster on some point x_i
                    print('to repeat',Dwin)
                    print('Dwin shape',Dwin.shape)
                    print(np.tile(Dwin,1,K.shape[1])-K)
                    #[tmp,new] = max(sum(max(repmat(Dwin,1,size(K,2))-K,0)));
                    k = k+1
                    #M = [M; L(new,:)+eps]
                    if pl:
                        print('new cluster, k = ',k)      
                    #[Dwin,Iwin] = min(Dist',[],2);
                    Dist = Dist.T
                    Dwin = np.amin(Dist,axis=1)
                    Iwin = np.argmin(Dist,axis=1)
                    #Wnew        = sum(Dwin);
                    Wnew = np.sum(Dwin,axis=0)
                    #Er=[Er; Wnew];
                    Er=np.vstack((Er, Wnew))
                    if len(T)!=0:
                        tmp = pairwise_distances(T,M,metric = "sqeuclidean")
                        TEr = np.vstack((TEr, np.sum(np.amin(tmp,axis=1),axis=0)))
            else:
                k=kmax+1
        #k=k+1
        Wold = Wnew
        if pl:
            #先畫X再畫Y
            plt.cla()
            ax.plot(X[:,2],X[:,1],'g.',M[:,2],M[:,1],'k+')
            fig.canvas.draw()
            time.sleep(0.00001)
    #RGB and BRG
    #Er=[Er; Wnew]
    Er = np.append(Er,Wnew)
    if len(T)!=0:
        #tmp=sqdist(T',M')
        #TEr=[TEr; sum(min(tmp,[],2))]
        TEr = np.concatenate((TEr,TEr),axis=0) 
        Er=np.concatenate((Er,TEr),axis=1) 
    #將M的第kill列刪去
    #M(kill-1,:)=[]
    M = np.delete(M,kill-1,axis=0)
    #for a in P:
    #    print(a)
    return [Er,M, nb, P]

if(__name__=='__main__'):
    img = np.load('Flower.npy')
    [m,n,d] = img.shape
    X = np.reshape(img,(m*n,d))
    [T,kmax,dyn,bs, killing, pl]=[[],16,0,0,0,1]
    kmeansO(X,T,kmax,dyn,bs, killing, pl,img)