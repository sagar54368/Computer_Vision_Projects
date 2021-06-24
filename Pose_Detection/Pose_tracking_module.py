import cv2
import mediapipe as mp
import time


class posedetector():

    def __init__(self, mode=False, upbody=False, smooth=True, detcon=0.5, trackcon=0.5):

        self.mode = mode
        self.upbody = upbody
        self.smooth = smooth 
        self.detcon = detcon
        self.trackcon = trackcon

        self.mpdraw = mp.solutions.drawing_utils
        self.mppose = mp.solutions.pose
        #
        self.pose = self.mppose.Pose(self.mode, self.upbody, self.smooth, self.detcon, self.trackcon)

    #method to find pose 
    def findpose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        #draw landmarks
        if self.results.pose_landmarks:
            if draw:
                self.mpdraw.draw_landmarks(img, self.results.pose_landmarks, self.mppose.POSE_CONNECTIONS)

        return img

    def getposition(self, img, draw=True):
        lmList = []
        #if results are available then use this for loop
        if self.results.pose_landmarks:
            #extract index of landmarks
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c =img.shape
                #print("id", id, "lm", lm)
                #pixel value of pts(landmarks)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                #draw circle
                if draw:                      
                    cv2.circle(img, (cx, cy), 5, (0,0 ,255), cv2.FILLED)
        
        return lmList



def main():
    cam = cv2.VideoCapture("__path__")
    ptime=0
    detector = posedetector()
    while True:
        success, img = cam.read()
        img = detector.findpose(img)
        lmList = detector.getposition(img, draw=False)

        if len(lmList) !=0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (255,0,0), cv2.FILLED)
        
        ctime= time.time()
        fps = 1/(ctime - ptime)
        ptime= ctime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()