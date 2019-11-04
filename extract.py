import os
import time
import pickle
import numpy as np
import face_recognition
import cv2

all_face_encondings = {}

files = os.listdir("dataset/")

for i in files:
    for j in os.listdir("dataset/"+i):
        all_face_encondings[i] = face_recognition.face_encodings(cv2.imread("dataset/"+i+"/"+j))
        print("[INFO] "+str(files.index(i))+"/"+str(len(files)))
'''
one_image = cv2.imread("dataset/Maxim/DSC_0117.jpg")
all_face_encondings["Maxim is the God"] = face_recognition.face_encodings(one_image)[0]

two_image = cv2.imread("dataset/Dmitrii/DSC_0118.jpg")
all_face_encondings["Dmitrii"] = face_recognition.face_encodings(two_image)[0]

three_image = cv2.imread("dataset/Egor/DSC_0119.jpg")
all_face_encondings["Egor"] = face_recognition.face_encodings(three_image)[0]

na_image = cv2.imread("dataset/Navalny/na.jpg")
all_face_encondings["Alexey Navalny"] = face_recognition.face_encodings(na_image)[0]

Putin_image = cv2.imread("dataset/Putin/Putin.jpg")
all_face_encondings["Vladimir Putin Molodets"] = face_recognition.face_encodings(Putin_image)[0]
'''
with open('dataset_faces_'+str(time.time())+'.dat', 'wb') as f:
    pickle.dump(all_face_encondings, f)
