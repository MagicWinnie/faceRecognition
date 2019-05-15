import face_recognition
import cv2
import numpy as np
import pickle
import logging

video_capture = cv2.VideoCapture(0)
with open('dataset_faces1.dat','rb') as ff:
    all_face_encodings=pickle.load(ff)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',filename='text1.log',level=logging.DEBUG)

known_face_names = list(all_face_encodings.keys())
known_face_encodings = np.array(list(all_face_encodings.values()))
count = 0
prev = ''
student = ['Dmitrii']
st=[0]
aas=0
aat=0
teacher = ['Maxim is the God']
te=[0]
names = []
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    if count%2==0:
        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            if len(names)<5:
                names.append(name)
            else:
                names = names[:4:]
                for i in student:
                    if i in names and aas!=1:
                        logging.debug(i + ' student in the frame')
                        aas=1
                    else:
                        if aas==1:
                            logging.debug(i + ' student disappeared')
                            aas=0
                        else:
                            logging.debug('unknown')
                for j in teacher:
                    if j in names and aat!=1:
                        logging.debug(j + ' teacher in the frame')
                        aat=1
                    else:
                        if aat==1:
                            logging.debug(j + ' teacher disappeared')
                            aat=0
                        else:
                            logging.debug('unknown')
                names.clear()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            prev = name

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    else:
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    count +=1
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()