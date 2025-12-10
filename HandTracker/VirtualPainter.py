import cv2
import HandTrackingModule as htm
import time
import os
import numpy as np

filepath = "Header"
filelist = os.listdir(filepath)
headerList = []
for i in filelist:
    img = cv2.imread(f'{filepath}/{i}')
    headerList.append(img)
header = headerList[0]

drawThickness = 15
eraserThickness = 35
drawColor = (255,0,0)
px,py = (0,0)
wCam, hCam = 1280,720
ptime = 0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)


detector = htm.handtracking(detect_con=1,track_con=1)
drawCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    net, frame = cap.read()
    frame = cv2.flip(frame, 1)

    detector.findhands(frame)

    lmlist = detector.findposition(frame,draw=False)

    if len(lmlist)!= 0:
        fingersUp = detector.fingersUp()

        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

        if fingersUp[1] and fingersUp[2] == True:
            px, py = (0, 0)

            cv2.rectangle(frame,(x1,y1-25),(x2,y2+25),(0,0,255),-1)
            if y1 < 125 :
                if 300<x1<360:
                    drawColor = (255,0,0)
                    header = headerList[0]

                elif 600<x1<670:
                    drawColor = (0,0,255)
                    header = headerList[1]

                elif 890<x1<950:
                    drawColor = (0,255,0)
                    header = headerList[2]

                elif 1150<x1<1220:
                    drawColor = (0,0,0)
                    header = headerList[3]

        if fingersUp[1] and fingersUp[2] == False:
            cv2.circle(frame,(x1,y1),10,(0,0,255),-1)
            if px == 0 and py == 0:
                px ,py = x1,y1

            if drawColor == (0,0,0):
                cv2.line(drawCanvas,(px,py),(x1,y1),drawColor,eraserThickness)

            else:
                cv2.line(drawCanvas,(px,py),(x1,y1),drawColor,drawThickness)

            px ,py = x1,y1

    img_gray = cv2.cvtColor(drawCanvas,cv2.COLOR_BGR2GRAY)
    _,img_Inv = cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY_INV)
    img_Inv = cv2.cvtColor(img_Inv,cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame,img_Inv)
    frame = cv2.bitwise_or(frame,drawCanvas)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    frame[0:125,0:1280] = header

    cv2.putText(frame,f'FPS: {int(fps)}',(5,25),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.imshow('Virtual Painter',frame)
    cv2.waitKey(1)

    if cv2.getWindowProperty('Virtual Painter',cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()