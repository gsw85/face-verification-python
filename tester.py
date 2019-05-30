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
from pyface import VGGFace, FaceDetector, FaceDescriptor, FaceClassifier, FaceVisualizer
from pyface.utils import *

from flask import Flask, render_template, Response
from camera import VideoCamera

WINDOW_NAME = 'FACE DEMO'

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


# def pipeline(frame, create=False, debug=True):
#     start = time.time()

#     frame = resize_scale(frame, 600, 600)

#     # Face Detection
#     frame, detected_faces = detection_pipeline(frame, visualize=True)

#     if not detected_faces['detected']: return frame

#     # Fece Description
#     face_descriptions = descriptor.get_resnet_descriptions(frame, detected_faces)
#     print(face_descriptions.shape)
#     # Face Classification
#     if create:
#         pass
#     else:
#         classifier.predict(face_descriptions)

#     if debug:
#         # {:>3} aligning strings to right
#         print('FPS: {:>3}'.format( int(1/(time.time()-start))) )

#     return frame

def test_detect_image():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # frame = cv2.imread(os.path.join('resources', 'test_images', 'selfies.jpg'))
    # frame = cv2.imread(os.path.join('resources', 'face_database', 'Poom', '1.jpg'))
    frame = cv2.imread(os.path.join('resources', 'face_database', 'Thada', '1.jpg'))

    detection_pipeline(frame, visualize=True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detection_pipeline(img, visualize=True):
    detected_faces = detector.predict(img, model='cnn')

    if detected_faces['detected'] and visualize:
        # visualizer.show_boxes(img, detected_faces['bounding_boxes'])
        visualizer.show_landmarks(img, detected_faces['landmark_points'])
        img = visualizer.show_triangles(img, detected_faces['landmark_points'])

    cv2.imshow(WINDOW_NAME, img)

    return img, detected_faces


if __name__ == "__main__":
    skil_service = None

    detector     = FaceDetector()
    descriptor   = FaceDescriptor(skil_service)
    classifier   = FaceClassifier()
    visualizer   = FaceVisualizer()

    test_detect_image()