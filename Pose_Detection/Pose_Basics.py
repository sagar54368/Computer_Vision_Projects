import cv2
import mediapipe as mp
import time

mpdraw = mp.solutions.drawing_utils
mppose = mp.solutions.pose
pose = mppose.Pose()
cam = cv2.VideoCapture("..path..")
#want to use own webcam use this
#cam = cv2.VideoCapture(0)
ptime=0

while True:
    success, img = cam.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)

    #draw landmarks
    if results.pose_landmarks:
        mpdraw.draw_landmarks(img, results.pose_landmarks, mppose.POSE_CONNECTIONS)
        #extract index of landmarks
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c =img.shape
            print(id, lm)
            #pixel value of pts(landmarks)
            cx, cy = int(lm.x*w), int(lm.y*h)
            #draw circle
            cv2.circle(img, (cx, cy), 5, (255,0 ,0), cv2.FILLED)


    ctime= time.time()
    fps = 1/(ctime - ptime)
    ptime= ctime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)