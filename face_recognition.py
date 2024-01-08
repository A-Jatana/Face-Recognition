import numpy as np
import cv2 as cv
import os # for path

haar_cascade = cv.CascadeClassifier('haar_face.xml') # read the haar cascade file and store it in a variable

people = []
DIR = r'C:\Users\amito\Face Recognition\Faces\train'
for i in os.listdir(DIR): 
    people.append(i)

features = np.load('features.npy', allow_pickle=True) # load the features array from the file
labels = np.load('labels.npy') # load the labels array from the file

face_recognizer = cv.face.LBPHFaceRecognizer_create() # create the face recognizer
face_recognizer.read('face_trained.yml') # read the trained model from the file

img = cv.imread(r'C:\Users\amito\Face Recognition\Faces\val\elton_john\3.jpg') # read the image

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert the image to grayscale
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4) # detect faces in the image

for (x,y,w,h) in faces_rect: # draw rectangles around the faces
    faces_roi = gray[y:y+h, x:x+h] # get the region of interest (the face) from the image

    label, confidence = face_recognizer.predict(faces_roi) # predict the label of the face
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2) # put the label on the image
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2) # draw a rectangle on the image

cv.imshow('Detected Face', img)

cv.waitKey(0)
