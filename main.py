import cv2
from RegionGrowing import RegionGrowing
import numpy as np

filename = '124084.jpg'
#filename = '892933215.jpg'
img = cv2.imread(filename)
cv2.imshow('flower',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
[Region1, Region2, Region3, Region4] = RegionGrowing(img)