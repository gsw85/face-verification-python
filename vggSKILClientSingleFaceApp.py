from keras.models import model_from_json, Sequential, Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, img_to_array, save_img
from keras.applications.imagenet_utils import preprocess_input
from  numpy.testing import assert_allclose
from PIL import Image
import shutil

import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join

import dlib
import cv2
import requests
import json

from numpy import ndarray
import urllib.request
import time
import sys
import uuid


# Documentation
# where to put the face image
# how to capture input image
# flip the image for front camera
# system configuration
# histEqualize
# use this as template for documentation

# get images of daily life. more emotions
# faces looking down is better

class VGGClientApp:

    similarityMinValue = 0.85
    imgRows = 224
    imgCols = 224
    channels = 3

    nameList = [] # name list
    nameEmbedding = dict() # name and the list of embeddings

    SKILServerPassword = "admin123"

    padHeight = 50
    padwidth = 20

    camera = 0 # default camera use 0
    inputImageHeight = -1
    inputImageWidth = -1

    # "http://localhost:9008//endpoints//vgg//model//vggFace//default//multipredict"
    embeddingURL      = 'http://localhost:9008/endpoints/vgg-deployment/model/vgg-model/v1/predict'
    kerasModelPath    = os.path.join("resources", "vgg_face_weights.h5")
    imageDbPath       = os.path.join("resources", "singleFaceDataset")
    embeddingDbPath   = os.path.join("resources", "singleFaceEmbedding")
    predictor_path    = os.path.join("resources", "shape_predictor_68_face_landmarks.dat")
    
    faceFontType      = cv2.FONT_HERSHEY_SIMPLEX
    faceFontScale     = 0.8
    faceFontColor     = (255, 0, 0)
    faceRectColor     = (0, 255, 0)
    faceRectThickness = 2


    def __init__(self, createDataBase = False):

        # get Face Detector
        self.detector = dlib.get_frontal_face_detector()

        # get FaceRecognizer local, to generate embeddings for database faces
        # when createDataBase set to True
        self.initDescriptor()

        # to get upright face for better embedding generation
        self.landmark_predictor = dlib.shape_predictor(self.predictor_path)

        # get Minimum Face Width for Face Verification
        self.minimumFaceWidth = 50 # int(self.imgCols / 4)

        self.initCamera()

        if createDataBase == True:
            # align each face from database, and save it
            self.createDatabase()

        else:
            self.loadDatabase()

        # get endpoint token
        self.getToken()

    def saveFaceForDatabase(self, mainPath):

        name = ""

        while True:
            name = input("What name is this new person>")
            name = name.lower() # lower case a string
            if(len(name) > 0):
                break

        pathToSave = mainPath + name
        files = os.listdir(self.imageDbPath)

        checkNameExist = False
        for name_exist in files:

            if name == name_exist:
                checkNameExist = True
            else:

                pathToDelete = self.imageDbPath + name_exist
                try:
                   print('Delete folder of path ' + pathToDelete)
                   shutil.rmtree(pathToDelete)
                except:
                   print('Error while deleting directory {}'.format(pathToDelete))

        if checkNameExist == False:
            os.mkdir(pathToSave)

        if(self.cap.isOpened() == False):
            self.cap = cv2.VideoCapture(self.camera)

        count = 0
        maxFacesCapture = 30

        while(True):
            # Capture frame-by-fram
            ret, frame = self.cap.read()

            frame = cv2.flip(frame, 1)

            # 4. Pass de loaded image to the `detector`
            dets = self.detector(frame, 0)

            # Display the resulting frame
            faces = dlib.full_object_detections()

            for detection in dets:
                # Find the 5 face landmarks we need to do the alignment.
                faces.append(self.landmark_predictor(frame, detection))

            if (len(dets) > 0):

                faces_aligned = dlib.get_face_chips(frame, faces, size = self.imgCols)

                rectList = self.getFacesRectangleList(dets)

                faceRecogIndex = self.getBiggestFaceIndex(dets)

                if faceRecogIndex != -1:
                    savePath = pathToSave + "\\" + str(uuid.uuid4()) + ".jpg"

                    det = rectList[faceRecogIndex]
                    # [top, bottom, topleft, right] to [topleft, top, right, bottom]
                    cv2.rectangle(frame, (det[2], det[0]), (det[3], det[1]), self.faceRectColor, self.faceRectThickness)
                    cv2.putText(frame, "Collect Face: %d  left" % (maxFacesCapture - count), (10, 85), self.faceFontType, 0.95, (255, 0, 0))
                    cv2.imshow("Saving Image For Database", frame)


                    cv2.imwrite(savePath, faces_aligned[faceRecogIndex])
                    count += 1


                    if(count > maxFacesCapture):
                        break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    # load all the embeddings of different faces in the rootPath
    # the directory can only contain subdirs. each of a particular person
    def createDatabase(self):

        print("Create embedding database ...")

        while True:
            choice = input("Do you want to captures faces for database [Y/N]>")

            if (choice == 'y') |(choice == 'Y') :

                databasePath = self.imageDbPath

                self.saveFaceForDatabase(self.imageDbPath)
                break
            else:

                break


        try:
           shutil.rmtree(self.embeddingDbPath)
        except:
           print('Error while deleting directory {}'.format(self.embeddingDbPath))

        # create embedding path if not there
        if(os.path.exists(self.embeddingDbPath) == False):
            endIndex = int(len(self.embeddingDbPath) - 1)
            embeddingFolder = self.embeddingDbPath[0:endIndex]
            os.mkdir(embeddingFolder)
            print("Directory " + embeddingFolder + " created")

        subDirs = [x[0] for x in os.walk(self.imageDbPath)]

        subDirs.remove(self.imageDbPath) # remove own path

        if len(subDirs) > 1:
             raise Exception("There should be one folder in {}.The folder name will be person name".format(self.imageDbPath))

        imagePath = [x[0] for x in os.walk(subDirs[0])][0]

        # get name which the the title of the folder
        name = imagePath.split("\\")[-1]

        self.nameList.append(name)
        self.nameEmbedding[name] = []

        print("Name: " + name)

        onlyfiles = [f for f in listdir(imagePath) if isfile(join(imagePath, f))]


        for image in onlyfiles:

            imageFullPath = imagePath + "\\" + image

            arr = self.createEmbeddingForDatabase(imageFullPath)
            if arr is not None: # -1 for no face found
                self.nameEmbedding[name].append(arr)


        df = pd.DataFrame(self.nameEmbedding[name])
        savePath = self.embeddingDbPath + name + ".txt"
        np.savetxt(savePath, self.nameEmbedding[name])

    def loadDatabase(self):

        print("Load embedding database")

        self.nameList = []

        files = os.listdir(self.embeddingDbPath)

        # to prevent more than one face.txt in the same folder
        txtFileCount = 0

        for file in files:
            if file.endswith(".txt"):

                if(txtFileCount > 1):
                     raise Exception("Number of files in the following folder {} should have only one txt file".format(self.embeddingDbPath))

                df_restored = np.loadtxt(os.path.join(self.embeddingDbPath, file))

                name = file[0:len(file) - 4]
                self.nameList.append(name)
                self.nameEmbedding[name] = df_restored

                txtFileCount = txtFileCount + 1

    def createEmbeddingForDatabase(self, imagePath):

        img = cv2.imread(imagePath, 1)
        # img = self.colorEqualizeHist(img)

        # grayImg = load_img(imagePath, grayscale = True, target_size = (self.imgRows, self.imgCols))

        dets = self.detector(img, 0) # img_to_array(img) second argument is for upscaling to detect smaller faces 0 for no upscaling

        # Display the resulting frame

        faces = dlib.full_object_detections()

        for detection in dets:
            # Find the 5 face landmarks we need to do the alignment.
            faces.append(self.landmark_predictor(img, detection))

        if(len(faces) == 1):

            face_aligned = dlib.get_face_chips(img, faces, size = self.imgRows)

            # cv2.imwrite('temp\\facecrop' + str(self.count) + ".jpg", face_aligned[0])
            # self.count = self.count + 1

            return self.getLocalMachineEmbedding(face_aligned[0])

        else:

            print("Skipped. No face found in this image.")
            return None

    def setInputImageSize(self, height, width):
        self.inputImageHeight = height
        self.inputImageWidth = width

    def initCamera(self):
        self.cap = cv2.VideoCapture(self.camera)

         # get vcap property
        width = self.cap.get(3)   # float
        height = self.cap.get(4) # float

        self.setInputImageSize(height, width)

        print("Image width: " + str(width))
        print("Image height: " + str(height))

    def getToken(self):
        # "http://localhost:9008/ui/
        # /login?returnUrl=%2Fworkspaces"
        loginURL = "http://localhost:9008/login"
        headers = {"Content-Type":"application/json", "Accept":"application/json"}
        payload = {
            'sysparm_action': 'insert',
            'short_description': 'test_jsonv2',
            'priority': '1'
        }
        payload = {'userId': 'admin', 'password': self.SKILServerPassword}
        response = requests.request("POST", loginURL, data=json.dumps(payload), headers = headers)

        if response.status_code != 200:
            raise Exception("Login to SKIL Server failed. Could not get token")
        else:
            loginJSON = response.json()
            for key, value in loginJSON.items():
                self.token = value
                print("Token: " + self.token)

        self.embeddingHeaders = {"Content-Type" : "application/json; charset=utf-8", "authorization" : "Bearer " + str(self.token)}

    def colorEqualizeHist(self, colorImage):

        img_yuv = cv2.cvtColor(colorImage, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def getMax(self, dictList):

        cosineMaxOutput = dict()

        for name in self.nameList:

            cosineMaxOutput[name] = max(dictList[name])

        return cosineMaxOutput

    def getTopAverage(self, dictList):

        top = 3
        cosineTopAverageOutput = dict()

        for name in self.nameList:

            if len(dictList[name]) > top:
                dictList[name].sort(reverse = True)

                sum = 0
                for i in range(0, top):

                    sum = sum + dictList[name][i]

                cosineTopAverageOutput[name] = sum / top
            else:
                print("database for particular person fewer than %d image" % top)

                cosineTopAverageOutput[name] =  self.getMax(dictList)

        return cosineTopAverageOutput

    def getLocalMachineEmbedding(self, faceImage):

        modelInput = self.preprocess_image(faceImage)

        embedding = self.vgg_face_descriptor.predict(modelInput)[0,:]

        return embedding

    def preprocess_image(self, image):
        # img = load_img(imagePath, target_size = (self.imgRows, self.imgCols))
        img = img_to_array(image)
        img = np.expand_dims(img, axis = 0) # add one more []
        # model accept batch size, this make single image ready to input into model
        img = preprocess_input(img)
        return img

    def getSingleFaceForEndpoint(self, currentFaceImage):
        # Expected input shape is height,width,channels,batch size
        # DL4j: input shape is batch size, channels, height, width
        # permute/transpose order is: 3,2,1,0 for inputs
        # permute/transpose order output:  3,2,1,0
        # currentFaceImage = load_img(currentFaceImage, target_size = (self.imgRows, self.imgCols))
        currentFaceImage_ = img_to_array(currentFaceImage)
        currentFaceImage_ = np.expand_dims(currentFaceImage_, axis=0) # add one more []
        currentFaceImage_ = preprocess_input(currentFaceImage_)

        currentFaceImage_ = np.array(currentFaceImage_)
        currentFaceImage_ = currentFaceImage_.ravel().tolist()

        imgEndpointInput = np.array(currentFaceImage_)
        imgEndpointInput = imgEndpointInput.reshape(1, self.imgRows, self.imgCols, self.channels)
        imgEndpointInput = np.transpose(imgEndpointInput, (0, 3, 1, 2))
        imgEndpointInput = list(imgEndpointInput.flatten())

        return imgEndpointInput

    def getEndpointEmbedding(self, faceInputVector):

        prediction_json = {
            "ordering":"c",
            "shape":(1, self.channels, self.imgRows, self.imgCols), 
            "data": faceInputVector
        }
        payload = {
            "needsPreProcessing":"false",
            "id":"25baeee4-2736-4135-8dfd-0ec168bd1976",
            "prediction": prediction_json
        }

        response = requests.request("POST", self.embeddingURL, data=json.dumps(payload), headers=self.embeddingHeaders)
        if response.status_code != 200:
            print('Status:', response.status_code, 'Headers:', response.headers, 'Error Response:', response.json())
            print(response.content)
            print(response.text)
            return []
        else:
            # print('response success')
            # print(response.status_code)
            json_result = response.json()
            response_data = json_result['prediction']
            embeddings = response_data['data']
            return embeddings

    def getMatchFace(self, singleFaceEmbedding):


        cosineNameDict = dict()

        for name in self.nameList:
            cosineNameDict[name] = []
            embeddingList = self.nameEmbedding[name]

            for dbEmbedding in embeddingList:

                cosineValue = self.getCosineSimilarity(dbEmbedding, singleFaceEmbedding)

                cosineNameDict[name].append(cosineValue)


                if(cosineValue > 0) & (cosineValue < 1):

                    cosineNameDict[name].append(cosineValue)

                else:

                    cosineNameDict[name].append(0)

        currentOutput = self.getTopAverage(cosineNameDict) # self.getTopAverage(cosineNameDict)

        return currentOutput

    def getBiggestFaceIndex(self, detection):

        biggestFaceIndex = -1 # FACE TOO SMALL FOR FACE RECOGNITION
        biggestFaceArea = self.minimumFaceWidth
        currentIndex = 0

        for det in detection:

            # row and column similar. Chose column as the standard of comparison
            currentFaceWidth = det.bottom() - det.top()

            if(currentFaceWidth > biggestFaceArea):
                biggestFaceArea = currentFaceWidth
                biggestFaceIndex = currentIndex

            currentIndex = currentIndex + 1

        return biggestFaceIndex

    def getFacesRectangleList(self, facesDect):

        facesRectangle = []

        for faceDect in facesDect:
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))

            topleft = 0
            top = 0
            bottom = faceDect.bottom()
            right = 0# int(width)


            if(faceDect.left() - self.padwidth) > 0:
               topleft =  faceDect.left() - self.padwidth

            if(faceDect.right() + self.padwidth) < self.inputImageWidth:
                right =  faceDect.right() + self.padwidth

            if(faceDect.top() - self.padHeight) > 0:
                top = faceDect.top() - self.padHeight

            faceRectangle = [top, bottom, topleft, right]

            facesRectangle.append(faceRectangle)

        return facesRectangle

    def runLocalFaceRecognition(self):

        if(self.cap.isOpened() == False):
            self.cap = cv2.VideoCapture(self.camera)

        while(True):
            # Capture frame-by-fram
            ret, frame = self.cap.read()

            frame = cv2.flip(frame, 1)

            # 4. Pass de loaded image to the `detector`
            dets = self.detector(frame, 0)

            # Display the resulting frame
            faces = dlib.full_object_detections()

            for detection in dets:
                # Find the 5 face landmarks we need to do the alignment.
                faces.append(self.landmark_predictor(frame, detection))

            if (len(dets) > 0):

                faces_aligned = dlib.get_face_chips(frame, faces, size = self.imgCols)

                rectList = self.getFacesRectangleList(dets)

                faceRecogIndex = self.getBiggestFaceIndex(dets)

                if faceRecogIndex != -1:
                    # endpoint
                    singleFaceEmbedding = self.getLocalMachineEmbedding(faces_aligned[faceRecogIndex])

                    dictOutput = self.getMatchFace(singleFaceEmbedding)

                    outputNameList = list(dictOutput.keys())

                    outputNameSimilarity = list(dictOutput.values())
                    maxValue = max(outputNameSimilarity)
                    maxIndex = int(outputNameSimilarity.index(maxValue))

                    if self.checkFaceConfident({outputNameList[maxIndex]: maxValue}):

                        cv2.putText(frame, "%s %.2f" % (outputNameList[maxIndex], outputNameSimilarity[maxIndex] * 100), (rectList[maxIndex][3], rectList[maxIndex][0]), self.faceFontType, self.faceFontScale, self.faceFontColor)# (right, top + padY), font, fontScale, fontColor)



                for det in rectList:
                    # [top, bottom, topleft, right] to [topleft, top, right, bottom]
                    cv2.rectangle(frame, (det[2], det[0]), (det[3], det[1]), self.faceRectColor, self.faceRectThickness)

                cv2.rectangle(frame, (rectList[faceRecogIndex][2], rectList[faceRecogIndex][0]), (rectList[faceRecogIndex][3], rectList[faceRecogIndex][1]), (0, 0, 255), self.faceRectThickness)

                cv2.imshow('Endpoint Face Verification', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

    def runEndpointFaceRecognition(self):

        # if(self.cap.isOpened() == False):
            # self.cap = cv2.VideoCapture(self.camera)

        self.cap = cv2.VideoCapture()
        # Opening the link
        self.cap.open("http://10.0.1.8:8080/video?.mjpeg")    

        while(True):
            # Capture frame-by-fram
            ret, frame = self.cap.read()
            
            frame = cv2.flip(frame, 1)

            # 4. Pass de loaded image to the `detector`
            dets = self.detector(frame, 0)

            # Display the resulting frame
            faces = dlib.full_object_detections()

            for detection in dets:
                # Find the 5 face landmarks we need to do the alignment.
                faces.append(self.landmark_predictor(frame, detection))

            if (len(dets) > 0):

                faces_aligned = dlib.get_face_chips(frame, faces, size = self.imgCols)

                rectList = self.getFacesRectangleList(dets)

                faceRecogIndex = self.getBiggestFaceIndex(dets)

                if faceRecogIndex != -1:
                    # endpoint
                    # currentFaceImage = self.colorEqualizeHist(faces_aligned[faceRecogIndex])
                    singleFaceForEndpoint = self.getSingleFaceForEndpoint(faces_aligned[faceRecogIndex])
                    singleFaceEmbedding = self.getEndpointEmbedding(singleFaceForEndpoint)

                    dictOutput = self.getMatchFace(singleFaceEmbedding)

                    if( self.checkFaceConfident(dictOutput) ):

                        padY = 0

                        for name in dictOutput.keys():

                            cv2.putText(frame, "%s %.2f" % (name, dictOutput[name]), (rectList[faceRecogIndex][3], rectList[faceRecogIndex][0] + padY), self.faceFontType, self.faceFontScale, self.faceFontColor)# (right, top + padY), font, fontScale, fontColor)

                            padY += 20


                for det in rectList:
                    # [top, bottom, topleft, right] to [topleft, top, right, bottom]
                    cv2.rectangle(frame, (det[2], det[0]), (det[3], det[1]), self.faceRectColor, self.faceRectThickness)

                cv2.rectangle(frame, (rectList[faceRecogIndex][2], rectList[faceRecogIndex][0]), (rectList[faceRecogIndex][3], rectList[faceRecogIndex][1]), (0, 0, 255), self.faceRectThickness)

                cv2.imshow('Endpoint Face Verification', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

    def checkFaceConfident(self, dictNameSimilarity):

        for name, similarityValue in dictNameSimilarity.items(): # assume length = 1
            if(similarityValue > self.similarityMinValue):
                return True
            else:
                return False

    def initDescriptor(self):

        self.model = Sequential()
        self.model.add(ZeroPadding2D((1,1), input_shape=(self.imgRows, self.imgCols, self.channels)))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(Convolution2D(4096, (7, 7), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Convolution2D(4096, (1, 1), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Convolution2D(2622, (1, 1)))
        self.model.add(Flatten())
        self.model.add(Activation('softmax'))
        self.model.load_weights(self.kerasModelPath)

        self.vgg_face_descriptor = Model(inputs = self.model.layers[0].input, outputs = self.model.layers[-2].output)
        # self.vgg_face_descriptor.save('model.h5')

    # value ranging from -1 to  1 . -1 as totally ot similar to 1 as totally similar
    def getCosineSimilarity(self, source, target):
        a = np.matmul(np.transpose(source), target)
        b = np.sum(np.multiply(source, source))
        c = np.sum(np.multiply(target, target))
        return (a / (np.sqrt(b) * np.sqrt(c)))

if __name__ == '__main__':

    faceClientApp = VGGClientApp(createDataBase = True)

    print("run Face Recognition Endpoint")
    faceClientApp.runEndpointFaceRecognition()
    # faceClientApp.runLocalFaceRecognition()

    print('End of program...')