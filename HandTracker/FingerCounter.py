import cv2
import time
import os
import HandTrackingModule as htm

filepath = "Fingerimages"
mylist = os.listdir(filepath)
overlaylist = []
for overimg in mylist:
    img = cv2.imread(f'{filepath}/{overimg}')
    img = cv2.resize(img, (200, 200))
    overlaylist.append(img)

wCam,hCam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
ptime = 0
detector = htm.handtracking(detect_con= 0.8,track_con=0.8)
fingerIds = [4,8,12,16,20]
while True:
    net, frame = cap.read()
    detector.findhands(frame,draw=True)
    lmlist = detector.findposition(frame,draw = False)
    if len(lmlist) != 0 :
        fingers = []
        # Thumb

        if lmlist[fingerIds[0]][1] > lmlist[fingerIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # four fingers

        for id in range(1, 5):
            if lmlist[fingerIds[id]][2] < lmlist[fingerIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        fingerCount = fingers.count(1)

        h, w, c = overlaylist[fingerCount-1].shape
        frame[280:h + 280, 440:w + 440] = overlaylist[fingerCount-1]

        cv2.rectangle(frame,(440,50),(620,250),(0,255,0),-1)
        cv2.putText(frame,f'{fingerCount}',(480,200),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),20)



    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(frame,f'FPS: {int(fps)}',(5,25),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow('Finger Counter', frame)

    cv2.waitKey(1)
    if cv2.getWindowProperty('Finger Counter',cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()