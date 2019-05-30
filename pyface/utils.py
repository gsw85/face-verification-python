import os
import cv2
import numpy as np

from keras.preprocessing.image import load_img, img_to_array, save_img
from keras.applications.vgg16 import preprocess_input

from skil import Skil, get_service_by_id


def get_skil_service():

    skil_server = Skil(
        host          = 'localhost',
        port          = 9008,
        user_id       = 'admin',
        password      = 'Skymind'
    )

    experiment_id   = 'vgg-experiment-01'
    model_id        = 'vgg-model-01'
    deployment_name = 'vgg-deployment'

    deployments = skil_server.api.deployments()
    deployment = next(deployment for deployment in deployments if deployment.name == deployment_name)
    deployment_id = deployment.id

    skil_service = get_service_by_id(
        skil_server,
        experiment_id,
        model_id,
        deployment_id
    )

    return skil_service



def video_streaming(video_source, scale=0.5):
    cap = cv2.VideoCapture(video_source)

    while(True):
        _, frame = cap.read()

        image_array = np.array(frame)
        
        image_array = resize_image(image_array, scale=scale)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('frame', image_array)

    cap.release()
    cv2.destroyAllWindows()


def load_image_to_array(image_path):

    image = load_img(image_path, target_size=(224, 224))

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # (1, 224, 224, 3) -> (1, 3, 224, 224)
    image_array = np.transpose(image_array, (0, 3, 1, 2))

    return image_array

# =============================== resize ====================================

def resize_image(image, scale=0.5):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA) 


def resize_scale(img, height_limit=1000, width_limit=1000):
    height, width, _ = img.shape

    height_ratio = height_limit/height
    width_ratio = width_limit/width

    scale = min(height_ratio, width_ratio)

    return(cv2.resize(img, (int(scale*width), int(scale*height))))

# ==============================================================================

"""
    Transform 68 facial landmarks shape to numpy array
"""
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# def argsort_face_area(detected_faces):
#     areas = []   
#     for box in detected_faces['bounding_boxes']:
#         y_range = (box['ymax'] - box['ymin'])
#         x_range = (box['xmax'] - box['xmin'])
#         areas.append(x_range*y_range)

#     # descending order
#     area_rank = np.argsort(areas)[::-1][:n]
#     return area_rank

# def get_top_n_face_area(self, detected_faces, areaRank, n):       
#     return {k: [v[i] for i in areaRank[:2]] for k, v in faces.items() if isinstance(v, list)}