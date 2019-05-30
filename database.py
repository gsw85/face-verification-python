import os
import sys
import json
import time
import dlib
import cv2
import pandas as pd
import numpy as np

from glob import glob
from keras.preprocessing.image import load_img, img_to_array, save_img
from keras.applications.imagenet_utils import preprocess_input

# Local application imports
from pyface import FaceDetector
from pyface import FaceDescriptor

FACE_DB_PATH = os.path.join("resources", "face_database")
FACE_EMBEDING_PATH = os.path.join("resources", "face_database", "encoded_faces.csv")

def sort_face_area(detected_faces):
    areas = []   
    for box in detected_faces['bounding_boxes']:
        y_range = (box['ymax'] - box['ymin'])
        x_range = (box['xmax'] - box['xmin'])
        areas.append(x_range*y_range)
    area_rank = np.flip(np.argsort(areas), axis=0)
    return area_rank

def get_top_face(detected_faces, n=1):
    area_rank = sort_face_area(detected_faces)
    top_faces = dict()
    for key, value in detected_faces.items():
        if isinstance(value, dlib.full_object_detections):
            # maintain full_object_detections class
            # e.g.,
            # From: [<dlib.full_object_detection object at 0x7f2f1dbb8228>]
            # TO: <dlib.full_object_detections object at 0x7f2f1dbb8298>
            top_faces[key] = dlib.full_object_detections(
                [value[i] for i in area_rank[:n]]
            )
        elif isinstance(value, list):
            top_faces[key] = [value[i] for i in area_rank[:n]]

    return top_faces

if __name__ == '__main__':
    # required instances
    skil_service = None
    detector     = FaceDetector()
    descriptor   = FaceDescriptor(skil_service)

    image_paths = glob(os.path.join(FACE_DB_PATH, '*', '*.jpg'))

    encoded_faces = []
    for path in image_paths:
        print(path)
        img = cv2.imread(path)
        # Face Detection
        detected_faces = detector.predict(img, model='cnn')

        if not detected_faces['detected']: continue

        detected_faces = get_top_face(detected_faces)
        # Fece Description
        face_description = list(descriptor.get_resnet_descriptions(img, detected_faces)[0])

        person_name = os.path.basename(os.path.dirname(path))
        encoded_faces.append(
            [person_name]+face_description
        )

    df_faces = pd.DataFrame.from_records(encoded_faces)
    df_faces.to_csv(FACE_EMBEDING_PATH, index=None)