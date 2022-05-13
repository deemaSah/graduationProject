# import the opencv library
import cv2
from imutils.video import VideoStream
import numpy as np
import imutils
import pickle
import time
import cv2
import os
# define a video capture object
vid = cv2.VideoCapture(0)
z=0
list=[]
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imwrite("images\img"+z.__str__() +".jpg", frame)
    list.append("images\img"+z.__str__() +".jpg")
    z=z+1
    ############################################################################################
    detector_path = './face_detection_model'
    embedding_model = './openface_nn4.small2.v1.t7'
    recognizer_path = './output/recognizer.pickle'
    le_path = './output/le.pickle'
    Confidence = 0.5

    protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
    modelPath = os.path.sep.join([detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    caffe_model = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    torch_model = cv2.dnn.readNetFromTorch(embedding_model)
    svm_model = pickle.loads(open(recognizer_path, "rb").read())
    le = pickle.loads(open(le_path, "rb").read())

    # img = cv2.imread("sample_image.jpg")  # from camera

    # هون بدنا نضيف ياخد سكرين شوت من الكاميرا يخزن الصورة وهي الي بعمل عليها بروسسيسنج وبدل كاميرا اللاب توب كاميرا المراقبة
    # list = ["sample_image.jpg"]  # list of images from camera
    x = 0
    for i in list:
        image = cv2.imread(i)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        # cv2.imwrite("copy"+x+".jpg", image)
        x += 1
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        caffe_model.setInput(imageBlob)
        detections = caffe_model.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > Confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                torch_model.setInput(faceBlob)
                vec = torch_model.forward()
                preds = svm_model.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                # show the output frame
                # cv2.imshow("Frame", image)
                cv2.imwrite("copy\copy" + x.__str__() + ".jpg", image)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
