import cv2
import numpy as np

img = cv2.imread('WIN_20251127_11_02_34_Pro.jpg')
img = cv2.resize(img,(0,0),fx=.5,fy=.5)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,100,.2,25)
corners = np.int_(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),10,(0,0,255),1)
cv2.imshow('gray',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
