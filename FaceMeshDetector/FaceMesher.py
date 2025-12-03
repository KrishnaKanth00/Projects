import cv2
import mediapipe as mp
import time


class FaceMeshDetector:

    def __init__(self,mode = False,noofface = 1,det_con = 0.5,trac_con=0.5):
        self.mode = mode
        self.num_faces = noofface
        self.det_con = det_con
        self.trac_con = trac_con
        self.mpmesh = mp.solutions.face_mesh
        self.mesh = self.mpmesh.FaceMesh( self.mode, max_num_faces = self.num_faces, min_detection_confidence = self.det_con,
                                         min_tracking_confidence = self.trac_con)
        self.mpdraw = mp.solutions.drawing_utils
        self.drawspec = self.mpdraw.DrawingSpec(color=(0,255,0),thickness=1,circle_radius=1)

    def findfacemesh(self,img):
        imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.mesh.process(imgrgb)

        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                self.mpdraw.draw_landmarks(img,facelms,self.mpmesh.FACEMESH_TESSELATION,self.drawspec,self.drawspec)

    def getposition(self,img):
        h,w,c = img.shape
        lmlist = []
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                face = []
                for id,lm in enumerate(facelms.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    face.append([id,x,y])
                lmlist.append(face)
        return lmlist



def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    faceMesher = FaceMeshDetector()
    while True:
        net,frame = cap.read()

        faceMesher.findfacemesh(frame)
        lmlist = faceMesher.getposition(frame)
        if len(lmlist) != 0:
            print(lmlist[0])
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(frame,f'FPS:{int(fps)}',(5,30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        cv2.imshow('Face Mesher',frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()