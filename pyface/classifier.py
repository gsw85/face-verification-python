import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

class FaceClassifier:

    SIMILARITY_THRESHOLD = 0.90
    FACE_EMBEDING_PATH = os.path.join("resources", "face_database", "encoded_faces.csv")

    def __init__(self):
        self.load_embedded_faces()
        

    def load_embedded_faces(self):
        df_faces = pd.read_csv(self.FACE_EMBEDING_PATH)

        print('Loaded: {} Embedded Faces'.format(len(df_faces)))

        # columns are [name + 128 features]
        self.X = df_faces[df_faces.columns[1:]].values
        self.y = df_faces[df_faces.columns[0]].values

        print(self.X.shape, self.y.shape)

        self.fit()

    def add_embedded_faces(self, face_description, person_name):
        df_faces = pd.read_csv(self.FACE_EMBEDING_PATH)

        face_description = list(face_description)

        print(person_name)
        print(face_description)

        df_adding = pd.DataFrame.from_records([
            [person_name]+face_description
        ])
        df_adding.columns = df_adding.columns.astype(str)
        print(df_adding)

        df_new = pd.concat([
            df_faces,
            df_adding
        ], axis=0)

        print(df_new)

        df_new.to_csv(self.FACE_EMBEDING_PATH, index=None)

        # columns are [name + 128 features]
        self.X = df_new[df_new.columns[1:]].values
        self.y = df_new[df_new.columns[0]].values

        print(self.X.shape, self.y.shape)

        self.fit()

    def fit(self):
        self.knn = NearestNeighbors(
            n_neighbors=1,
            metric='cosine',
            n_jobs=1
        ).fit(self.X)

    def predict(self, face_descriptions):
        recognized_faces = dict()
        distances, indices = self.knn.kneighbors(face_descriptions, return_distance=True)
        for ind, (distance, index) in enumerate(zip(distances, indices)):
            # since n_neighbors=1, it returns only one neighbor.
            cosine = 1-distance[0]
            index = index[0]

            # print(cosine)

            if cosine < self.SIMILARITY_THRESHOLD:
                recognized_faces[ind] = 'Unknown'
            else:
                recognized_faces[ind] = self.y[index]
            
        return recognized_faces
