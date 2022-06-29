import cv2
# define a video capture object
vid = cv2.VideoCapture('rtsp://192.168.1.133:554/profile2')
vid2 = cv2.VideoCapture(0)
z=0
u=0
list=[]
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    ret2, frame2 = vid2.read()
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('frame2', frame2)
    cv2.imwrite("images\imgA"+z.__str__() +".jpg", frame)
    list.append("images\imgA"+z.__str__() +".jpg")
    cv2.imwrite("images\imgB" + z.__str__() + ".jpg", frame2)
    list.append("images\imgB" + z.__str__() + ".jpg")
    z=z+1