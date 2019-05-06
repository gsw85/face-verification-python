# Face Verification / Recognition using SKIL endpoint and local machine

## Overview 

Face Verification Using VGG Model and cosine similarity function for small amount of face images.

Landmark file and VGG Model is stored in 
https://drive.google.com/open?id=1LbaliUYMlQseXz1i3KJPfj16pye12rtl

Use case overview written in English can be found here. https://drive.google.com/open?id=1vU5Xg675mf_VKA3kEiUCRSsKgmOezKzM6OqKIsHDCaY

Use case overview written in mandarin can be found here. https://drive.google.com/open?id=1uDmd3VtVmfL2_RqD51lPgv1fBF6N_CFXdIlwm-wmrHE

## Deployment Environment 
The program was developed and tested in windows environment. 
Below are the version environment. 
Python 3.7.2
Packages: dlib, keras, tensorflow / tensorflow-gpu, numpy, urllib, pandas

## Files and Folders
**vggSKILServerDeploymentNotebook.txt**

This is the file for SKIL Server model deployment. When this program to tested on SKIL 1.2.1, uploading json file to SKIL is facing problem. Hense the using of txt format. Each %pyspark separation in the txt file is a cell in SKIL Zeppelin Notebook

**vggSKILClientSingleFaceApp.py**

This is the client applicatioon for SKIL Client. It reaches out to SKIL endpoint for embedding vectors and compare with the database to find matching ids for the face. This only recognize the face with the biggest surface area (all the faces in the frame is still detected) 

**vggSKILClientMultiFacesApp.py**

Similar to above. However, this file detects and recognizes all faces in a frame. 

## How to run

**SKIL Server side**
Import vggSKILServerDeploymentNotebook.txt manually and run on SKIL Workspace. Make sure model is started on SKIL Deployment

**SKIL Client side**
Run vggSKILClientSingleFaceApp or vggSKILClientMultiFacesApp.

*faceClientApp = VGGClientApp(createDataBase = True)*
set createDataBase = True for the first time to generate embeddings in resources\facesEmbedding or resources\singleFaceEmbedding
Also, set it to True to get images from camera input

*faceClientApp.runEndpointFaceRecognition()*
Use this function to get embedding vector from endpoint.
Set the embeddingURL as the deployed endpoint on SKIL

*faceClientApp.runLocalFaceRecognition()*
Use this function to get embedding vector from the same program

## How to add new faces data 

(vggSKILClientSingleFaceApp) To add new faces, add images into a folder with the name of the person in resources\singleFaceDataset

(vggSKILClientMultiFacesApp) To add new faces, add images into a folder with the name of the person in resources\facesDataset

For example, resources\singleFaceDataset\john

In john folder, there are 1.jpg, 2.jpg, 3.jpg and etc. 
