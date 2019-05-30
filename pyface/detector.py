"""
    Face Detection Model
"""
import os
import dlib

from .utils import shape_to_np

class FaceDetector:

    FACE_SIZE = 224
    CNN_FACE_DETECTOR_PATH = os.path.join("resources", "models", "mmod_human_face_detector.dat")
    LANDMARK_PREDICTOR_PATH = os.path.join("resources", "models", "shape_predictor_68_face_landmarks.dat")

    def __init__(self):
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(self.CNN_FACE_DETECTOR_PATH)
        self.landmark_predictor = dlib.shape_predictor(self.LANDMARK_PREDICTOR_PATH)

    """
        get cropped & aligned faces 
    """
    def predict(self, frame, model=''):
        if model is 'cnn':
            detections = self.cnn_face_detector(frame)
        else:
            detections = self.hog_face_detector(frame)
         
        
        # detections not empty
        if detections:
            
            face_objects = dlib.full_object_detections()
            landmark_points = []
            bounding_boxes = []
        
            # find face landmarks to do the alignment.
            for detection in detections:

                if model is 'cnn': detection = detection.rect
                
                # face landmarks
                landmarks = self.landmark_predictor(frame, detection)
                landmark_points.append(shape_to_np(landmarks))

                # store face landmarks to full_object_detections
                face_objects.append(landmarks)
                
                # extract bounding boxes
                bounding_boxes.append(self.get_bounding_boxes(detection))

            # rotate upright and scale to FACE_SIZE pixels
            aligned_faces = dlib.get_face_chips(frame, face_objects, size=self.FACE_SIZE)

            
            return {
                'detected': True,
                'aligned_faces': aligned_faces,
                'bounding_boxes': bounding_boxes,
                'landmark_points': landmark_points,
                'face_objects': face_objects
            }

        return {'detected': False}

    """
        Get Face Bounding Boxes
    """
    def get_bounding_boxes(self, detection):
        return {
            "ymin": detection.top(),
            "ymax": detection.bottom(),
            "xmin": detection.left(),
            "xmax": detection.right()
        }