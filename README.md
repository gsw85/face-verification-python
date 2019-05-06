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
vggSKILServerDeploymentNotebook.txt: 
This is the file for SKIL Server model deployment. When this program to tested on SKIL 1.2.1, uploading json file to SKIL is facing problem. Hense the using of txt format. Each %pyspark separation in the txt file is a cell in SKIL Zeppelin Notebook

vggSKILClientSingleFaceApp.py
This is the client applicatioon for SKIL Client. It reaches out to SKIL endpoint for embedding vectors and compare with the database to find matching ids for the face. This only recognize the face with the biggest surface area (all the faces in the frame is still detected) 

vggSKILClientMultiFacesApp.py
Similar to above. However, this file detects and recognizes all faces in a frame. 

## How to run
## How to add new faces data 
