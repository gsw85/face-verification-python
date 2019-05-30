"""
    Face Recognition using VGG Face Descriptor
"""

# Standard library imports
import os
import sys
import json
import time

import glob
import cv2
import dlib
import numpy as np

# Local application imports
from pyface import VGGFace, FaceDetector, FaceDescriptor, FaceVisualizer
from pyface.utils import *

from flask import Flask, render_template, Response
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
visualizer   = FaceVisualizer()

def pipeline(frame, create=False, debug=True):
    start = time.time()

    frame = resize_scale(frame, 600, 600)

    # Face Detection
    frame, detected_faces = detection_pipeline(frame, visualize=True)

    if not detected_faces['detected']: return frame

    # Fece Description
    face_descriptions = descriptor.get_resnet_descriptions(frame, detected_faces)

    # Face Classification
    # if create:
    
    # else:
        

    if debug:
        # {:>3} aligning strings to right
        print('FPS: {:>3}'.format( int(1/(time.time()-start))) )

    return frame

    

def detection_pipeline(img, visualize=True):
    detected_faces = detector.predict(img, model='cnn')

    if detected_faces['detected'] and visualize:
        # visualizer.show_boxes(img, detected_faces['bounding_boxes'])
        visualizer.show_landmarks(img, detected_faces['landmark_points'])
        img = visualizer.show_triangles(img, detected_faces['landmark_points'])

    return img, detected_faces

def gen(camera):
    while True:
        frame = camera.get_frame()

        # pyFace Pipeline
        frame = pipeline(frame, create=False, debug=True)
        
        frame = camera.to_byte(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    video_source = 'http://10.0.1.8:8080/video?.mjpeg'
    return Response(
        gen(VideoCamera(video_source)),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)