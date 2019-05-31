"""
    Face Recognition using VGG Face Descriptor
"""

# Standard library imports
import os
import sys
import json
import time
import copy
import glob
import cv2
import dlib
import numpy as np

import subprocess
# Local application imports
from pyface import VGGFace, FaceDetector, FaceDescriptor, FaceClassifier, FaceVisualizer
from pyface.utils import *

from flask import Flask, render_template, Response, request
from camera import VideoCamera

# WINDOW_NAME = 'FACE DEMO'

# def test_detect_image():
#     frame = cv2.imread(os.path.join('resources', 'test_images', 'selfies.jpg'))

#     detection_pipeline(frame, visualize=False)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def test_detect_video(video_source):
#     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#     cap = cv2.VideoCapture(video_source)

#     while(True):
#         _, frame = cap.read()
        
#         frame = np.array(frame)
#         frame = resize_scale(frame)

#         detection_pipeline(frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'): 
#             break

#     cap.release()
#     cv2.destroyAllWindows()


app = Flask(__name__)

skil_service = None

detector     = FaceDetector()
descriptor   = FaceDescriptor(skil_service)
classifier   = FaceClassifier()
visualizer   = FaceVisualizer()

face_entities = dict()

def pipeline(frame, face_id, update=False, debug=True, person_name=None):
    start = time.time()
    frame = resize_scale(frame, 600, 600)

    # Face Detection
    detected_faces = detected_faces = detector.predict(frame, model='cnn')

    if not detected_faces['detected']:
        if face_id != 0:
            return np.full((224, 224), 250)
        return frame
    # Fece Description
    face_descriptions = descriptor.get_resnet_descriptions(frame, detected_faces)

    # Face Classification
    if update:
        classifier.add_embedded_faces(face_descriptions[0], person_name)
    
    recognized_faces = classifier.predict(face_descriptions)

    frame = visualizer.show_landmarks(
        frame, 
        detected_faces['landmark_points'],
        recognized_faces,
        triangles=False
    )

    if debug:
        # {:>3} aligning strings to right
        print('FPS: {:>3}'.format( int(1/(time.time()-start))))

    if face_id == 0:
        return frame
    elif face_id <= len(detected_faces['aligned_faces']):
        face = detected_faces['aligned_faces'][face_id-1]
        return face
    return np.full((224, 224), 250)

def gen(camera, face_id):
    while True:
        frame = camera.get_frame()
        # pyFace Pipeline
        frame = pipeline(frame, face_id, update=False, debug=True)
        frame = camera.to_byte(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.context_processor
def inject_face_id():
    return dict(face_entities=face_entities)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        person_name = request.form['person']

        person_dir = os.path.join('resources', 'face_database', person_name)
        if not os.path.exists(person_dir):
            os.mkdir(person_dir)

        print(person_name, file.filename)
        file.save(os.path.join(person_dir, file.filename))

        img = cv2.imread(os.path.join(person_dir, file.filename))
        pipeline(img, face_id=0, update=True, person_name=person_name)

    return render_template('index.html')

@app.route('/404')
def page404():
    return render_template('404.html')

@app.route('/video_feed/<int:face_id>')
def video_feed(face_id):
    # video_source = 'http://10.0.1.8:8080/video?.mjpeg'
    video_source = 'http://192.168.42.129:8080/video?.mjpeg'
    return Response(
        gen(VideoCamera(video_source), face_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)