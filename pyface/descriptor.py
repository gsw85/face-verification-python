import os

import dlib
import numpy as np

from keras.preprocessing.image import load_img, img_to_array, save_img
from keras.applications.vgg16 import preprocess_input


class FaceDescriptor:

    RESNET_PATH = os.path.join("resources", "models", "dlib_face_recognition_resnet_model_v1.dat")

    def __init__(self, skil_service):
        self.skil_service = skil_service
        self.resnet_dlib_descriptor = dlib.face_recognition_model_v1(self.RESNET_PATH)


# =========================== VGG Descriptor =============================

    def get_vgg_descriptions(self, detected_faces, predictor):
        # initial an empty array with shape
        face_descriptions = np.empty((1, 2622), int)

        for img in detected_faces['aligned_faces']:
            # preprocess image array
            img = self.transform_image_array(img)
            
            switcher = {
                'vgg_skil': self.predict_vgg_skil(img),
                'vgg_local': self.predict_vgg_local(img),
            }
            tmp = switcher.get(predictor)
            
            face_descriptions = np.append(face_descriptions, tmp, axis=0)

        return face_descriptions


    def predict_vgg_skil(self, img):
        # array of (1, 2622) face description/features
        return self.skil_service.predict(img)

    def predict_vgg_local(self, img):
        return 0

# ============================================================================
# ================================== RESNET Descriptor ===================================

    def get_resnet_descriptions(self, frame, detected_faces):
        # array of (n, 128) face description/features
        face_descriptions = self.resnet_dlib_descriptor.compute_face_descriptor(
            frame,
            detected_faces['face_objects']
        )
            
        return np.array(face_descriptions)

# ============================================================================


    def transform_image_array(self, img):
        # (224, 224, 3) to (1, 224, 224, 3)
        # Keras works with batches of images. 
        # So, the first dimension is used for the number of samples
        img = np.expand_dims(img, axis=0)

        # to adequate your image to the format the model requires
        img = preprocess_input(img)

        # (1, 224, 224, 3) -> (1, 3, 224, 224)
        img = np.transpose(img, (0, 3, 1, 2))

        return img