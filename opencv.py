import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray,100,.5,25)
    corners = np.int_(corners)

    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(frame,(x,y),10,(0,0,255),1)
    cv2.imshow('gray',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
