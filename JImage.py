import numpy as np
import time
from sklearn.metrics import pairwise_distances
from JCalculation import JCalculation

def JImage(I,W):
    [m,n] = I.shape
    ws = W.shape[0]

    if ws == 9:
        d = 1
    elif ws == 17:
        d = 2
    elif ws == 33:
        d = 4
    elif ws == 65:
        d = 8
    wswidth = ws // 2

    JI = np.zeros((m,n))
    for i in range(1,m+1):
        for j in range(1,n+1):
            print(i)
            x1 = i-wswidth
            x2 = i+wswidth
            y1 = j-wswidth
            y2 = j+wswidth
            if x1<1:
                x1 = 1
            if x2>m:
                x2 = m
            if y1<1:
                y1 = 1
            if y2>n:
                y2 = n
            wid = x2-x1+1
            hei = y2-y1+1
            if wid == ws and hei == ws:
                #median of the window, because of sampling, M won't change, neither St
                St = 1080
                M = [5, 5]
            else:
                reg = np.ones((wid, hei))
                reg = reg[0::d, 0::d]
                [wid, hei] = reg.shape
                M = [[np.mean(range(1,wid+1)), np.mean(range(1,hei+1))]]
                z = np.argwhere(reg)
                #sorting z using second column
                z = z[z[:,1].argsort(kind='stable')]
                z = z+1
                K=pairwise_distances(z,M,metric = "sqeuclidean")
                St = np.sum(K)

            block = np.zeros((ws, ws))
            block = I[x1:x2+1, y1:y2+1]
            Jval = JCalculation(block[0::d, 0::d], M, St)
            JI[i-1,j-1] = Jval
            #print(JI[i-1,j-1])
    return JI

if __name__ == '__main__':
    w1 = np.load('Window1.npy')
    w2 = np.load('Window2.npy')
    w3 = np.load('Window3.npy')
    w4 = np.load('Window4.npy')
    mapping = np.load('mapping.npy')
    JImage(mapping,w1)