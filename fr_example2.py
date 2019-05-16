import face_recognition
import cv2
import numpy as np
import pickle
import logging
face_locations = []
face_encodings = []
face_names = []
prev_names = []
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
print("Turned ON Camera")
with open('dataset_faces1.dat','rb') as ff:
    all_face_encodings=pickle.load(ff)
print("Loaded encodings")
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',filename='text2.log',level=logging.DEBUG)

known_face_names = list(all_face_encodings.keys())
known_face_encodings = np.array(list(all_face_encodings.values()))
count = 0

people = [['Dmitrii',0, 's'],['Putin',0, 's'],['Maxim is the God',0, 't'],['Alexey Navalny',0, 't']]

names = []
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    rgb_frame = frame[:, :, ::-1]
    if True:
        if count%2==0:
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
        if len(list(set(face_names)-set(prev_names)))>0:
            raz = list(set(face_names)-set(prev_names))
            print(raz)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
        
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)    
    count +=1
    prev_names=face_names.copy()
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()