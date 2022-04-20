from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
import cx_Oracle

# create connection
conStr='system/deema@localhost:1521/orcl'

conn = cx_Oracle.connect(conStr)
cur = conn.cursor()
########################################################

#set file's path
dataset ='./dataset'#*
embeddings = './output/embeddings.pickle'#*
detector_path = './face_detection_model'
model ='./openface_nn4.small2.v1.t7'
Confidence = 0.5
#############################################################
sqlTxt='select * from "SYSTEM"."EMPLOYEES" '
cur.execute(sqlTxt)
records = cur.fetchall()


#############################################################
# Download caffe & troch model
protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])#**
modelPath = os.path.sep.join([detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])
caffe_model = cv2.dnn.readNetFromCaffe(protoPath, modelPath)#Download a caffe model
troch_model = cv2.dnn.readNetFromTorch(model)#Download a Troch model
###############################################################
# training data
#imagePaths = list(paths.list_images(dataset))
trainingEmbeddings = []
trainingNames = []
arrEmpID=[]
numOfFaces = 0
###############################################################
#print(imagePaths[0])
# Extracting embed vectors for faces
for record in records:
    #name = imagePath.split(os.path.sep)[-2]# Extract the person's name from the path
    name = record[1]
    ID=record[0]
    query='select * from "SYSTEM"."TRAININGIMG" where EMP_ID =:1 '
    cur.execute(query,(record[0],))
    images= cur.fetchall()
    #n=1
    for img in images:
        for i in range(1, img.__len__()):
            image = cv2.imread(img[i])
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
            #cv2.imwrite("cut"+i.__str__()+".jpg", image)
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
            # print("i")
            # print(i)
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
                arrEmpID.append(ID)
                numOfFaces += 1




data = {"embeddings": trainingEmbeddings, "names": trainingNames}
#print(data)
f = open(embeddings, "wb")
f.write(pickle.dumps(data))
f.close()

for i in range(0, trainingNames.__len__()):
    query='INSERT INTO "SYSTEM"."EMBADDINGS" (EMBADDINGS, EMP_ID) VALUES (:1, :2)'
    cur.execute(query,(trainingEmbeddings[i].__str__(),arrEmpID[i],))





#print(trainingNames.__len__())

# saving data to mysql database
#db = mysql.connector.connect(host="localhost",user="root",passwd="root",database="face_recognetion")

#mycursor=db.cursor()
#sql = ("INSERT INTO data"
 #      "(embedding,name,ID)"
  #     "VALUES(%s,%s,%s)")
#u=0
#for i in trainingEmbeddings:

 #data2=(trainingEmbeddings[u].__str__(),trainingNames[u],0)
 #mycursor.execute(sql,data2)
 #db.commit()
 #u+=1
 #نضيف if وكمان كويري بجيب فيها التيبل وبعدها بفحص اذا الداتا موجودة من قبل عشان ما ارجع اضيفها
#mycursor.close()
#print(trainingEmbeddings[0])