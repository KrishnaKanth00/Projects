import cv2

cap = cv2.VideoCapture(0)
model = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN_create(model,"",(640,480))
while True:
    ret,frame = cap.read()
    detector.setInputSize((frame.shape[1], frame.shape[0]))

    faces = detector.detect(frame)

    if faces[1] is not None:
        for face in faces[1]:
            x,y,w,h = face[:4].astype(int)
            lx,ly,rx,ry = face[4:8].astype(int)

            cv2.circle(frame,(lx,ly),15,(0,255,0),2)
            cv2.circle(frame, (rx, ry), 15, (0, 255, 0), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)



    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()




