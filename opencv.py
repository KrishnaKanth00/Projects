import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while True:
    net, frame = cap.read()
    height = int(cap.get(4))
    width = int(cap.get(3))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([35, 40, 40])

    upper_gray = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame',res )
    cv2.imshow('mask',mask)



    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()