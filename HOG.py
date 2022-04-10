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
    cv2.imwrite("cut.jpg", image)



    # preprocess before passing it through our deep neural network for classification.
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # We assumed that the average pixel values of
    # the three layers are (123.0 , 177.0 , 104.0) because the Caffe model is a pre-trained model on the
    # ImageNet training dataset

    caffe_model.setInput(imageBlob)

    detections = caffe_model.forward()
   # print("detections")
#    print(detections[0, 0, :, 2])

    # Processing detected faces
    # ensure at least one face was found
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        #print("i")
        #print(i)
        confidence = detections[0, 0, i, 2]
        if confidence > Confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue


            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            troch_model.setInput(faceBlob)
            vec = troch_model.forward()
            trainingNames.append(name)
            trainingEmbeddings.append(vec.flatten())
            numOfFaces += 1
data = {"embeddings": trainingEmbeddings, "names": trainingNames}
f = open(embeddings, "wb")
f.write(pickle.dumps(data))
f.close()
