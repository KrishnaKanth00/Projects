import cv2
import mediapipe as mp
import time

class FaceDetector:

    def __init__(self,det_con = 0.5,model_sel = 0):

        self.det_con = det_con
        self.model_sel = model_sel
        self.mpface = mp.solutions.face_detection
        self.face = self.mpface.FaceDetection(det_con,model_sel)
        self.mpdraw = mp.solutions.drawing_utils

    def findface(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.face.process(imgRGB)
        bboxs = []
        if results.detections:
            for id,det in enumerate(results.detections):
                #self.mpdraw.draw_detection(img,det)
                bboxc = det.location_data.relative_bounding_box
                h,w,c = img.shape
                bbox = int(bboxc.xmin * w),int(bboxc.ymin * h), int(bboxc.width * w),int(bboxc.height * h)
                bboxs.append([id,bbox,det.score])

                if draw:
                    img = self.fancydraw(img,bbox)

                    cv2.putText(img, str(int(det.score[0] * 100)), (bbox[0], bbox[1] - 15), cv2.FONT_HERSHEY_PLAIN, 2,
                                (0, 255, 0),
                                3)

        return img ,bboxs

    def fancydraw(self,img,bbox,l = 15,lt = 5):

        x,y,w,h = bbox
        x1,y1 = x+w , y+h
        #topleft
        cv2.line(img,(x,y),(x,y+l),(255,0,0),lt)
        cv2.line(img,(x,y),(x+l,y),(255,0,0),lt)
        # top right
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 0), lt)
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 0), lt)
        # bottom  left
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 0), lt)
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 0), lt)
        # bottom right
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 0), lt)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 0), lt)


        cv2.rectangle(img, bbox, (255, 0, 0), 1)

        return img




def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    faceDetector = FaceDetector()
    while True:
        ret, frame = cap.read()
        frame , box = faceDetector.findface(frame)

        print(box)

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(frame,f'FPS: {int(fps)}',(20,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.imshow('Face Detector', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

