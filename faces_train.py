import os # for path
import cv2 as cv # for image processing
import numpy as np # for array manipulation

people = []
DIR = r'C:\Users\amito\Face Recognition\Faces\train'
for i in os.listdir(DIR): 
    people.append(i)

features = [] 
labels = []

def create_train():
    for person in people: 
        path = os.path.join(DIR, person) # get the path to the folder of the person
        label = people.index(person) # get the label of the person

        for img in os.listdir(path):
            img_path = os.path.join(path, img) # path to image in folder

            img_array = cv.imread(img_path) # read the image
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) # convert the image to grayscale

            haar_cascade = cv.CascadeClassifier('haar_face.xml') # read the haar cascade file and store it in a variable

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4) # detect faces in the image

            for (x,y,w,h) in faces_rect: # draw rectangles around the faces
                faces_roi = gray[y:y+h, x:x+w] # get the region of interest (the face) from the image
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object') # convert the features list to a numpy array
labels = np.array(labels) # convert the labels list to a numpy array


face_recognizer = cv.face.LBPHFaceRecognizer_create() # create the face recognizer
# LBPHFaceRecognizer is a face recognizer that uses the Local Binary Patterns Histograms algorithm

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml') # save the trained model to a file

np.save('features.npy', features) # save the features array to a file
np.save('labels.npy', labels) # save the labels array to a file
