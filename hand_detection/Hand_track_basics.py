import cv2 
import mediapipe as mp
import time

#using Mediapipe Solutions module
mpHands = mp.solutions.hands
hands = mp.solutions.hands.Hands()
#for drawing ponts and connectins btw points on  hands
mpdraw = mp.solutions.drawing_utils
#For video capturing using webcam
cam = cv2.VideoCapture(0)

#For Time and Frame rate
ptime = 0.0
ctime = 0.0
while True :
    success,img = cam.read()
    h, w, c = img.shape
    #MediaPipe only uses RGB images but Opencv here gives us BGR images so we convert it to RGB images
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    output = hands.process(imgRGB)

    #now we check if the output has multiple hands and try to get the landmarks of those hands
    if output.multi_hand_landmarks:
        for handsLM in output.multi_hand_landmarks:
            #Information for each hand in video using Landmark(gives index and X&Y coordinates)
            for id ,lm in enumerate(handsLM.landmark):
                #The model has 21 handmarks and this each landmarks id number along with X,y,z coordinates
                #The model returns decimal coordinates which are a ratio of height and width so we multiply with H and W to get Pixel VAlues and precise Location
                cx, cy = int(lm.x*w),int(lm.y*h) #Centre X and Y for each 21 Landmarks
                print("ID :",id,"Cx:",cx,"CY:",cy)
                """if id == 5: #By using this we can highlight any particle Landmk and use it 
                # cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)"""
            #for Drwaing joints and lines connecting those joints(21 Points of interest)
            mpdraw.draw_landmarks(img,handsLM,mpHands.HAND_CONNECTIONS)

    ctime = time.time()
    FPS = 1/(ctime - ptime)
    ptime = ctime
    fps = str(int(FPS)) 
    cv2.putText(img,str(fps),(10,100),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,255),1)
    
    
    cv2.imshow("Webcam",img)
    cv2.waitKey(1)