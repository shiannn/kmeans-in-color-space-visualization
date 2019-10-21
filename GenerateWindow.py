import numpy as np

def GenerateWindow(scale):
    window = np.ones((65, 65))
    #left up block of the window
    lu = np.ones((32,32))
    #left bottom
    lb = np.ones((32,32))
    #right up
    ru = np.ones((32,32))
    #right bottom
    rb = np.ones((32,32))

    #left up
    j = 0
    for i in range(24,10-2,-2):
        lu[j,0:i] = 0
        lu[0:i,j] = 0
        j = j + 1
    #90 degree counterclockwise rotation of matrix
    lb = np.rot90(lu)
    rb = np.rot90(lb)
    ru = np.rot90(rb)

    window[0:32, 0:32] = lu
    window[0:32, 33:65] = ru #33~64
    window[33:65, 0:32] = lb
    window[33:65, 33:65] = rb

    w4 = window
    w3 = w4[0::2, 0::2]
    w2 = w3[0::2, 0::2]
    w1 = w2[0::2, 0::2] #contain boundary

    if scale == 4: 
        W = w4
    elif scale == 3:
        W = w3
    elif scale == 2:
        W = w2
    elif scale == 1:
        W = w1

    return W
    
if __name__=='__main__':
    W = GenerateWindow(2)
    print(W)

    print(W.shape)