import pickle
import face_recognition
import cv2
import numpy as np

all_face_encondings = {}

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

one_image = cv2.imread("dataset/Boldysheva Ekaterina/IMG_20190513_135904.jpg")
all_face_encondings["Boldysheva Ekaterina"] = face_recognition.face_encodings(one_image)[0]

two_image = cv2.imread("dataset/Ostapenko Kirill/IMG_20190513_135742.jpg")
all_face_encondings["Ostapenko Kirill"] = face_recognition.face_encodings(two_image)[0]

three_image = cv2.imread("dataset/Balako Alexey/IMG_20190513_135807.jpg")
all_face_encondings["Balako Alexey"] = face_recognition.face_encodings(three_image)[0]

four_image = cv2.imread("dataset/Danilov Maxim/IMG_20190513_140057.jpg")
all_face_encondings["Danilov Maxim"] = face_recognition.face_encodings(four_image)[0]

five_image = cv2.imread("dataset/Griboedov Nikita/IMG_20190513_135642.jpg")
all_face_encondings["Griboedov Nikita"] = face_recognition.face_encodings(five_image)[0]

six_image = cv2.imread("dataset/Simakov Stepan/IMG_20190513_140017.jpg")
all_face_encondings["Simakov Stepan"] = face_recognition.face_encodings(six_image)[0]

seven_image = cv2.imread("dataset/Machalov Andrey/IMG_20190513_135834.jpg")
all_face_encondings["Machalov Andrey"] = face_recognition.face_encodings(seven_image)[0]

eight_image = cv2.imread("dataset/Parakhin Nikita/IMG_20190513_140033.jpg")
all_face_encondings["Parakhin Nikita"] = face_recognition.face_encodings(eight_image)[0]

nine_image = cv2.imread("dataset/Sherbakov Danil/IMG_20190513_135730.jpg")
all_face_encondings["Sherbakov Danil"] = face_recognition.face_encodings(nine_image)[0]

ten_image = cv2.imread("dataset/Sokolnikov Egor/IMG_20190513_135936.jpg")
all_face_encondings["Sokolnikov Egor"] = face_recognition.face_encodings(ten_image)[0]

lmm_image = cv2.imread("dataset/Makarov Alexander/IMG_20190513_135954.jpg")
all_face_encondings["Makarov Alexander"] = face_recognition.face_encodings(lmm_image)[0]

with open('dataset_faces1.dat', 'wb') as f:
    pickle.dump(all_face_encondings, f)
