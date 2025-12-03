import cv2
import time
import HandTrackingModule as htm
import math
from pycaw.pycaw import AudioUtilities
import numpy as np

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume

wCam ,hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
ptime = 0
detector = htm.handtracking(detect_con=0.9,track_con=0.8)
volRange = volume.GetVolumeRange()
min_vol = volRange[0]
max_vol = volRange[1]
volBar = 400
level = 0
prev_vol = 0
while True:
    net , frame = cap.read()

    detector.findhands(frame)
    lmlist = detector.findposition(frame,draw=False)
    if len(lmlist) != 0:

        tx,ty = lmlist[4][1],lmlist[4][2]
        ix,iy = lmlist[8][1],lmlist[8][2]
        cx,cy = int(ix+tx)//2 , int(iy+ty)//2


        cv2.circle(frame,(tx,ty),10,(0,255,0),-1)
        cv2.circle(frame,(ix,iy),10,(0,255,0),-1)
        cv2.circle(frame, (cx,cy), 10, (0, 255, 0), -1)

        cv2.line(frame,(tx,ty),(ix,iy),(255,0,255),3)

        length = math.hypot(ix-tx,iy-ty)

        vol = np.interp(length,(20,200),[min_vol,max_vol])
        volBar = np.interp(length,(20,200),[400,100])
        level = np.interp(length, (20, 200), [0, 100])

        # this moving average filter so that the change or transition would be smooth
        if abs(vol - prev_vol) > 1.0:  # threshold
            volume.SetMasterVolumeLevel(vol, None)
            prev_vol = vol

        if length < 50:
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

    cv2.rectangle(frame,(20,100),(60,400),(255,0,0),2)
    cv2.rectangle(frame,(20,int(volBar)),(60,400),(255,0,0),cv2.FILLED)
    cv2.putText(frame,f"{int(level)} %",(20,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)


    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(frame,f"FPS: {int(fps)}",(5,25),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow('Volume Controller',frame)
    cv2.waitKey(1)

    if cv2.getWindowProperty("Volume Controller",cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()
