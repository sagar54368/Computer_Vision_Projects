from flask import Flask, render_template, Response
import cv2

app=Flask(__name__)
cam = cv2.VideoCapture(0)

def generate_frames():
    while True:

        success,frame = cam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)  # encode to jpg format
            # convert this buffer back to bytes
            frame = buffer.tobytes()
        
        yield(b'--frame\r\n'       # use yield istead of simple return as return will return only 1 frame and stop
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)