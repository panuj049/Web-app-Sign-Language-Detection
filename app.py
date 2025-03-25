import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from flask import Flask, render_template, Response, request, redirect, url_for

app = Flask(__name__, static_folder='static')

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\Anuj Pawar\Downloads\Sign-Language-detection-main\Sign-Language-detection-main\Model\keras_model.h5",
                        r"C:\Users\Anuj Pawar\Downloads\Sign-Language-detection-main\Sign-Language-detection-main\Model\labels.txt")

# Constants for image preprocessing
offset = 20
imgSize = 300
labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

capturing = False

def generate_frames():
    global capturing
    cap = cv2.VideoCapture(0)
    while True:
        if not capturing:
            continue

        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0 and imgCrop.shape[0] != 0 and imgCrop.shape[1] != 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capturing
    capturing = True
    return '', 204

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global capturing
    capturing = False
    return '', 204

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
