
import os
import sys
import json

import glob
import cv2
import pandas as pd
import numpy as np

class FaceDatabase:

    FACE_DB_PATH = os.path.join("..", "resources", "face_database")

    def argsort_face_area(self, detected_faces):
        areas = []   
        for box in detected_faces['bounding_boxes']:
           y_range =  (box['ymax'] - box['ymin'])
           x_range =  (box['xmax'] - box['xmin'])
           areas.append(x_range*y_range)
        area_rank = np.flip(np.argsort(areas))
        return area_rank
    
    def get_top_n_face_area(self, detected_faces, area_rank, n): 
        top_faces = dict()
        for key, value in detected_faces.items():
            if isinstance(value, list):
                top_faces[key] = [value[i] for i in area_rank[:n]]
        return top_faces

