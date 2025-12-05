import cv2
import mediapipe as mp
import time
import math

class PoseDetector():
    def __init__(self,mode = False,smooth_lm = True,dect_con = 0.5,trac_con = 0.5):
        self.mode = mode
        self.smooth_lm = smooth_lm
        self.dect_con = dect_con
        self.trac_con = trac_con
        self.mppose = mp.solutions.pose
        self.pose = self.mppose.Pose()
        self.mpdraw = mp.solutions.drawing_utils


    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpdraw.draw_landmarks(img, self.results.pose_landmarks, self.mppose.POSE_CONNECTIONS)

    def findPosition(self,img,draw = True):
        h, w, c = img.shape
        self.lmlist = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                cx,cy = int(lm.x*w),int(lm.y*h)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),10,(255,0,0),3)
        return self.lmlist

    def findangle(self,img,p1,p2,p3,draw = True):
        x1,y1 = self.lmlist[p1][1:]
        x2,y2 = self.lmlist[p2][1:]
        x3,y3 = self.lmlist[p3][1:]

        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

        if angle < 0:
            angle += 360
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 2)
            cv2.circle(img,(x1,y1),5,(0,0,255),-1)
            cv2.circle(img,(x1,y1),10,(0,0,255),2)

            cv2.circle(img,(x2,y2),5,(0,0,255),-1)
            cv2.circle(img,(x2,y2),10,(0,0,255),2)

            cv2.circle(img,(x3,y3),5,(0,0,255),-1)
            cv2.circle(img,(x3,y3),10,(0,0,255),2)

            cv2.putText(img,str(int(angle)),(x2 - 30,y2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)

        return angle



def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    poseDetector = PoseDetector()
    while True:
        net, frame = cap.read()
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        poseDetector.findPose(frame)

        lm = poseDetector.findPosition(frame,draw = False)
        if len(lm) != 0:
            print(lm[4])
        cv2.putText(frame, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 3)
        cv2.imshow('Pose Tracking', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__" :
    main()



 