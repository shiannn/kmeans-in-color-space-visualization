import numpy as np
from kmeansO import kmeansO
from GenerateWindow import GenerateWindow

def RegionGrowing(img):
    [m,n,d] = img.shape
    X = np.reshape(img,(m*n,d))
    X = X.astype(np.float64)
    #[tmp,M,tmp2,P] = kmeansO(X,[],16,0,0,0,0,img)
    
    tmp = np.load('DistanceSumBetweenPointAndCluster.npy') #tmp是
    M = np.load('cluster16_RGB.npy') #M是16個cluster中心，每個都有 RGB 3個數值
    #tmp2不會用到
    P = np.load('PointBelongToCluster.npy')#P是一個 154401 x 1 的向量，每個的值都是0~15，代表每個點隸屬於哪個cluster
    mapping = np.reshape(P,(m, n))
    print(mapping.shape)

    for w in range(1,5):
        W = GenerateWindow(w)
        #JI{w} = JImage(map, W)  #cell array   

    return [0,0,0,0]

if __name__ == '__main__':
    #filename = '124084.jpg'
    #img = cv2.imread(filename)
    img = np.load('Flower.npy')
    [Region1, Region2, Region3, Region4] = RegionGrowing(img)