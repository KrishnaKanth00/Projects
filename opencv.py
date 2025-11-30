import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
prev_time = 0
curr_time = 0


mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    net , frame = cap.read()
    rgbImg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgbImg)

    if results.multi_hand_landmarks is not None:
        for handsLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame,handsLms,mphands.HAND_CONNECTIONS)


    curr_time = time.time()
    fps = 1/ (curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(frame,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()



