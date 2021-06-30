import cv2
import Hand_tracking_module as ht
import time
import os

# wCam , hCam = 640, 480

cam = cv2.VideoCapture(0)
# cam.set(3, wCam)
# cam.set(4, hCam)


# 
folderPath = "Finger"
mylist = os.listdir(folderPath)
print(mylist)
overlaylist = []

for i in mylist:
    image = cv2.imread(f'{folderPath}/{i}')
    # print(f'{folderPath}/{i}')
    overlaylist.append(image)

ptime = 0
detector = ht.handdetector(detectionCon=0.75)

tipids = [4, 8, 12, 16, 20]

while True:
    success, img = cam.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    
    fingers = []
    if len(lmList)!=0:
        #print value of list at any index(landmark)
        #print(lmList)
        # Thumb
        if lmList[tipids[0]][1] > lmList[tipids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)  

        
        for id in range(1, 5):
            if lmList[tipids[id]][2] < lmList[tipids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)      

        #print(fingers)
    totalfingers = fingers.count(1)
    print(totalfingers)     

    
    # slicing img[range of height, range of width]
    h, w, c = overlaylist[totalfingers - 1].shape
    img[0:h, 0:w] = overlaylist[totalfingers - 1]

    cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalfingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 25)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (440, 80), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3,
                    (255, 0, 255), 3)

    cv2.imshow("Image",  img)
    cv2.waitKey(1)