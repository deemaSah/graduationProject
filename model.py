# import the necessary packages
import xlsxwriter
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
import cx_Oracle
from openpyxl import workbook,load_workbook
from sklearn.svm import SVC

#create file
#print(ws)


# create connection
conStr='system/deema@localhost:1521/orcl'

conn = cx_Oracle.connect(conStr)
cur = conn.cursor()
########################################################
#path to output serialized db of facial embeddings
embeddings = './output/embeddings.pickle'

#path to OpenCV's deep learning face detector
detector_path = './face_detection_model'

#path to OpenCV's deep learning face embedding model
embedding_model ='./openface_nn4.small2.v1.t7'

#minimum probability to filter weak detections
Confidence = 0.5
#############################################################
sqlTxt='select * from SYSTEM."TDATA" '
cur.execute(sqlTxt)
records = cur.fetchall()
conn.commit()
cur.close()
conn.close()
#############################################################



# load our serialized face detector from disk
protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)



# load our serialized face embedding model from disk
embedder = cv2.dnn.readNetFromTorch(embedding_model)

# grab the paths to the input images in our dataset
#imagePaths =list(paths.list_images(dataset))
imagePaths=[]
# corresponding people names
knownEmbeddings = []
knownNames = []
#imagePaths.append(records[1][3])
print(len(records))
for record in records :
	imagePaths.append(record[3])
	knownNames.append(record[2])
	#print(record)
# initialize the total number of faces processed
total = 0
print(imagePaths)
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
		if confidence > Confidence:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# add the name of the person + corresponding face
			# embedding to their respective lists
			#knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1
			#print(imagePath)
#print(knownNames)
#print(imagePaths)

# dump the facial embeddings + names to disk
#print("[INFO] serializing {} encodings...".format(total))
#print(knownNames)
i=0
j=2
conn = cx_Oracle.connect(conStr)
cur = conn.cursor()
print(records)
for rec in records:
	query = ' UPDATE SYSTEM."TDATA" SET EMBADDINGS =:1 WHERE EMB_ID=:2 '
	cur.execute(query,[knownEmbeddings[i].__str__(),rec[0]])
	#wb = load_workbook("C:\\Users\\HP\\OneDrive\\Documents\\TED\\test.xlsx")
	#ws = wb.active
	#print(len(knownEmbeddings[0]))
	i=i+1
	#j=j+1
	#wb.save("C:\\Users\\HP\\OneDrive\\Documents\\TED\\test.xlsx")

#cur.execute(sqlTxt)

conn.commit()
cur.close()
conn.close()
###################################################
#outSheet.write("A1","ID")
#outSheet.write("A2","2")

#for emb in knownEmbeddings :
#	print(emb)

data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(embeddings, "wb")
f.write(pickle.dumps(data))
f.close()
#################################################################
# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
#recognizer.s