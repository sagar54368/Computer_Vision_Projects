from flask import Flask, render_template, Response
import cv2
import Hand_tracking_module as ht
import time
import os


app=Flask(__name__)
cam = cv2.VideoCapture(0)

def generate_frames():
    
    folderPath = "Finger"
    mylist = os.listdir(folderPath)
    print(mylist)
    overlaylist = []

    for i in mylist:
        image = cv2.imread(f'{folderPath}/{i}')
        # print(f'{folderPath}/{i}')
        overlaylist.append(image)

    ptime = 0
    detector = ht.handdetector(detectionCon=0.5)

    tipids = [4, 8, 12, 16, 20]


    while True:

        success,img = cam.read()
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

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', img)  # encode to jpg format
            # convert this buffer back to bytes
            img = buffer.tobytes()
        
        yield(b'--img\r\n'       # use yield istead of simple return as return will return only 1 frame and stop
                       b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=img')


if __name__=="__main__":
    app.run(debug=True)

