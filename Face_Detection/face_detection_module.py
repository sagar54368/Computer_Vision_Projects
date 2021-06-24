import cv2
import mediapipe as mp
import time

class facedetector():
    def __init__(self, mindetectioncon=0.5):
        
        self.mindetectioncon = mindetectioncon
        
        #using face detection module of mediapipe
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.mindetectioncon) #min_detection confidence (default = 0.5)

    
    def findfaces(self,img, draw= True):

        #convert to rgb
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.output  = self.faceDetection.process(imgRGB)
        #print(self.output)
        
        # for storing values of id, score and location of box
        bboxs = []
        #for multiple faces detection
        if self.output.detections:
            for id, detection in enumerate(self.output.detections):
                #extract info of bounding box
                # bounding box contain x_min, y_min, h, w od box
                bboxC = detection.location_data.relative_bounding_box  #these are normalised valurs(bet 0 & 1)
                # convert in own lang (pixel values)
                h, w, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)

                bboxs.append([id, bbox, detection.score])
                
                # Draw rectangle
                #cv2.rectangle(img, bbox, (255,0,255), 2)

                if draw:
                    img = self.fancyDraw(img, bbox)

                    #for printing score
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return img, bboxs

   
    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        x, y, w1, h1 = bbox
        x1, y1 = x + w1, y + h1

        cv2.rectangle(img, bbox, (255, 255, 0), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (0, 255, 0), t)
        cv2.line(img, (x, y), (x, y+l), (0, 255, 0), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y+l), (0, 255, 0), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 0), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 255, 0), t)
        
        return img



def main():
    cam = cv2.VideoCapture("S:/PROJECTS/videos/v2.mp4")
    #cam = cv2.VideoCapture(0)
    ptime = 0
    detector = facedetector()

    while True:
        success, img = cam.read()

        img, bboxs = detector.findfaces(img)
        print(bboxs)

        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    
        cv2.imshow("Image", img)
        cv2.waitKey(10)



if __name__ == "__main__":
    main()