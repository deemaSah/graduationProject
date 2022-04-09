from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

#set file's path
dataset ='./dataset'
embeddings = './output/embeddings.pickle'
detector_path = './face_detection_model'
model ='./openface_nn4.small2.v1.t7'
Confidence = 0.5

# Download caffe & troch model
protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])
caffe_model = cv2.dnn.readNetFromCaffe(protoPath, modelPath)#Download a caffe model
troch_model = cv2.dnn.readNetFromTorch(model)#Download a Troch model

# training data
imagePaths = list(paths.list_images(dataset))
trainingEmbeddings = []
trainingNames = []
numOfFaces = 0

# Extracting embed vectors for faces
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]# Extract the person's name from the path
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]


# construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # We assumed that the average pixel values of
    # the three layers are (123.0 , 177.0 , 104.0) because the Caffe model is a pre-trained model on the
    # ImageNet training dataset
    caffe_model.setInput(imageBlob)
    detections = caffe_model.forward()
