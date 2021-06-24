import cv2
import mediapipe as mp
import time

#cam = cv2.VideoCapture("S:/PROJECTS/videos/v2.mp4")
cam = cv2.VideoCapture(0)

ptime = 0

#using face detection module of mediapipe
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mp.solutions.face_detection.FaceDetection(0.75) #min_detection confidence (default = 0.5)



while True:
    success, img = cam.read()

    #convert to rgb
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output  = faceDetection.process(imgRGB)
    print(output)

    #for multiple faces detection
    if output.detections:
        for id, detection in enumerate(output.detections):
            
            # raw box around face using points(=6)
            # Default drawinf fucn
            #mpDraw.draw_detection(img, detection)
            
            # print(id, detection)
            # accuracy of detections
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            
            #extract info of bounding box
            # bounding box contain x_min, y_min, h, w od box
            bboxC = detection.location_data.relative_bounding_box  #these are normalised valurs(bet 0 & 1)
            # convert in own lang (pixel values)
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                 int(bboxC.width * w), int(bboxC.height * h)

            # Draw rectangle
            cv2.rectangle(img, bbox, (255,0,255), 2)
            #for printing score
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)




    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    
    cv2.imshow("Image", img)
    cv2.waitKey(10)
