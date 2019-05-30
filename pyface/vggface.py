"""
    VGG Face Descriptor Model
"""

import os

from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation

class VGGFace:

    def __init__(self):
        self.model_path = os.path.join('..', 'resources', 'vgg_face_weights.h5')
        self.input_shape = (224, 224, 3)

    """
        VGG Face Architecture
    """
    def model(self):
        model = Sequential()

        # Block 1
        model.add(ZeroPadding2D((1, 1), input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 2
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 5
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 6
        model.add(Conv2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        model.load_weights(self.model_path)

        vgg_face_descriptor = Model(inputs=model.layers[0].input,
                                    outputs=model.layers[-2].output)

        return vgg_face_descriptor