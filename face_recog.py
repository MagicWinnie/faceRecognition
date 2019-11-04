import time
import face_recognition
import cv2
import numpy as np
import pickle

#video_capture = cv2.VideoCapture("videos/test_out_04.avi")
video_capture = cv2.VideoCapture(0)
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

with open('dataset_faces1.dat','rb') as ff:
    all_face_encodings=pickle.load(ff)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(str(time.time())+'.avi',fourcc, 29.97, (1920,1080))

known_face_names = list(all_face_encodings.keys())
known_face_encodings = np.array(list(all_face_encodings.values()))

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    if not ret:
        break
    
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

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

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    out.write(frame)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == 27:
        break


out.release()
video_capture.release()
cv2.destroyAllWindows()
