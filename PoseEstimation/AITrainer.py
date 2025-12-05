import cv2
import time

import numpy as np

import PoseEstimation


wCam ,hCam = 1280,720
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
detector = PoseEstimation.PoseDetector(dect_con=0.8,trac_con=0.8)

ptime = 0
dir = 0
count = 0
bar = 0
per = 0

while True:
    net, frame = cap.read()
    detector.findPose(frame,draw = False)
    lmlist = detector.findPosition(frame,draw = False)
    color = (0, 255, 0)
    if len(lmlist) != 0:

        angle = detector.findangle(frame,12,14,16)
        per = np.interp(angle,(40,130),(100,0))
        bar = np.interp(angle,(40,130),(1000,300))


        if per == 0:
            if dir == 0:
                count += .5
                dir = 1
        if per == 100:
            color = (0, 0, 255)
            if dir == 1:
                count += .5
                dir = 0


        cv2.rectangle(frame,(0,550),(180,720),(0,255,0),-1)
        cv2.putText(frame, str(int(count)), (40, 670), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
        cv2.rectangle(frame,(300,650),(1000,700),(0,255,0),2)
        cv2.rectangle(frame,(300,650),(int(bar),700),color,-1)
        cv2.putText(frame, f'{str(int(per))}%', (1050,700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(frame,f'FPS: {int(fps)}',(5,25),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.imshow('AI Trainer',frame)
    cv2.waitKey(1)

    if cv2.getWindowProperty('AI Trainer',cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()