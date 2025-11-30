import cv2
import time

cap = cv2.VideoCapture(0)

prev_time = 0
curr_time = 0

while True:
    net , frame = cap.read()

    curr_time = time.time()
    fps = 1/ (curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(frame,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()



